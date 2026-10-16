"""Control-variate accuracy estimators for M_16,10 under ARC's protocol (m16/arc_eval.py interface).

Layers (each optional):
  1. Rao-Blackwellisation over the last input (rb9.Q): exact P(correct | x0..x8), charged COST_Q evals.
  2. Stratification of the prefix x0..x8 by model-free near-tie strata (exact probabilities, model-free).
  3. Control variate: cross-fitted linear regression of Q on model-free prefix features (basis9) whose
     per-stratum means are known model-free. Unbiased: coefficients for fold k are fit on the other fold.
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", ".."))
from rb9 import Q, COST_Q
from basis9 import basis9

T = 10
EDGES = np.array([0.0, 0.01, 0.025, 0.05, 0.1, 0.2, np.inf])
NS = len(EDGES)


def pstratum(x):
    s = -np.sort(-x[:, :9], 1); m1, m2, m3 = s[:, 0], s[:, 1], s[:, 2]
    g = np.minimum((m1 - m2) / np.abs(m1), (m2 - m3) / np.maximum(np.abs(m2), 1e-12))
    b = 1 + np.clip(np.searchsorted(EDGES, g, side="right") - 1, 0, NS - 2)
    return np.where(m2 <= 0, 0, b)


PRE = os.path.join(HERE, "precomp.npz")


def precompute(n=20_000_000, chunk=500_000, seed=123, kinds="AB"):
    """Model-free: stratum probabilities and per-stratum feature means (no model evaluations)."""
    rng = np.random.default_rng(seed); cnt = np.zeros(NS); sums = {k: None for k in kinds}; tot = {k: None for k in kinds}
    for _ in range(n // chunk):
        x = rng.standard_normal((chunk, 9)); h = pstratum(x); cnt += np.bincount(h, minlength=NS)
        for k in kinds:
            B = basis9(x, k)
            S = np.stack([B[h == j].sum(0) for j in range(NS)])
            sums[k] = S if sums[k] is None else sums[k] + S
            tot[k] = B.sum(0) if tot[k] is None else tot[k] + B.sum(0)
    out = {"P": cnt / n}
    for k in kinds:
        out["MU_" + k] = sums[k] / cnt[:, None]; out["M_" + k] = tot[k] / n
    np.savez(PRE, **out)


_pc = None
def pc():
    global _pc
    if _pc is None:
        _pc = dict(np.load(PRE))
    return _pc


def draw(h, k, rng, batch=100_000):
    out, got = [], 0
    while got < k:
        x = rng.standard_normal((batch, 9)); x = x[pstratum(x) == h]; out.append(x); got += len(x)
    return np.concatenate(out)[:k]


def _cf_cv(q, Bc, w, ridge=1e-3):
    """Cross-fitted (2 folds) CV-corrected values q - c.Bc with weights w (sum over samples = target)."""
    n = len(q); idx = np.random.default_rng(len(q)).permutation(n); folds = [idx[: n // 2], idx[n // 2:]]
    out = q.copy()
    for a, b in ((0, 1), (1, 0)):
        fa, fb = folds[a], folds[b]
        W = w[fa] / w[fa].sum(); X = Bc[fa]; y = q[fa]
        xm = (W[:, None] * X).sum(0); ym = (W * y).sum()
        Xc = X - xm; yc = y - ym
        G = (Xc * W[:, None]).T @ Xc; g = (Xc * W[:, None]).T @ yc
        G += ridge * np.trace(G) / G.shape[0] * np.eye(G.shape[0])
        c = np.linalg.solve(G, g)
        out[fb] = q[fb] - Bc[fb] @ c
    return out


def make(strat=False, cv=None, alloc=None, cost=COST_Q):
    """Build est(model, budget, rng). cv in {None,'A','B'}."""
    def est(model, budget, rng):
        n = int(budget // cost)
        P = pc()["P"]
        if strat:
            nh = np.maximum(4, np.floor(alloc * n).astype(int))
            xs = [draw(h, nh[h], rng) for h in range(NS)]
            x = np.concatenate(xs); h = np.repeat(np.arange(NS), nh)
            w = (P / nh)[h]
        else:
            x = rng.standard_normal((n, 9)); h = np.zeros(n, int); w = np.full(n, 1.0 / n)
        q = Q(model, x)
        if cv is not None:
            B = basis9(x, cv)
            Bc = B - (pc()["MU_" + cv][pstratum(x)] if strat else pc()["M_" + cv][None])
            q = _cf_cv(q, Bc, w)
        return float((w * q).sum()), int(np.ceil(len(x) * cost))
    return est
