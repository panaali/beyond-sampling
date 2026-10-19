"""T291 screen: propagate Hermite coefficients of the hidden states (config S: d=3, k=8), count FLOPs as executed,
then the pre-registered gates and diagnostics (README). Usage: python run.py [d k] ...

Deductive part: `propagate` (weights -> coefficients, no random inputs). Everything with `rng` below is a labelled
DIAGNOSTIC Monte Carlo (never part of an estimate).
"""
import json
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, HERE)
from herm import Algebra  # noqa: E402

# arc_eval imports m16/common.py (torch + alg_zoo, original weights). Not installed here: inject a stub module whose
# arrays are only used by noisy_models (never called). The harness's forward() is used unchanged.
_stub = types.ModuleType("common")
_stub.Wi, _stub.Wh, _stub.Wo = np.ones(16), np.eye(16), np.ones((10, 16))
sys.modules["common"] = _stub
sys.path.insert(0, os.path.join(ROOT, "m16"))
import arc_eval  # noqa: E402

FPE = 5948.0
SCREEN = ["fused_s0", "fused_s1", "torch_s0", "torch_s1"]
T, H = 10, 16


def load_panel():
    z = np.load(os.path.join(HERE, "panel/dev2_weights.npz"))
    names = list(z["names"])
    truths = json.load(open(os.path.join(HERE, "panel/dev2_truths.json")))["truths"]
    return z, names, truths


def cheb_relu(a, b, k):
    """Monomial coefficients (in y = (u - c)/r, y in [-1,1]) of the degree-k Chebyshev interpolant of ReLU on [a,b].
    Model-dependent only through (a, b): O(k^2) FLOPs, charged."""
    c, r = (a + b) / 2, (b - a) / 2
    j = np.arange(k + 1)
    y = np.cos(np.pi * (j + .5) / (k + 1))
    f = np.maximum(c + r * y, 0)
    cc = np.polynomial.chebyshev.chebfit(y, f, k)
    return np.polynomial.chebyshev.cheb2poly(cc), c, r, 4 * (k + 1) ** 2


def propagate(model, A, k, width=4.0):
    """Deductive: Hermite coefficients p_t (H, M) of every hidden state. Returns (P list, flops, log)."""
    Wi, Wh, Wo = model
    M = A.M
    p = np.zeros((H, M))
    xs = [A.var(t) for t in range(T)]
    flops = 0.0
    P, log = [], []
    for t in range(T):
        act = A.maxvar < t + 1                       # indices on x_1..x_{t+1}
        ma = int(act.sum())
        u = Wh @ p + np.outer(Wi, xs[t])
        flops += 2 * H * H * ma + 2 * H
        newp = np.zeros_like(p)
        kinds = []
        for i in range(H):
            mu = u[i, 0]
            sd = np.sqrt((u[i, 1:] ** 2 * A.norm2[1:]).sum()); flops += 3 * ma
            a, b = mu - width * sd, mu + width * sd
            if a >= 0:
                newp[i] = u[i]; kinds.append("lin"); continue
            if b <= 0:
                kinds.append("zero"); continue
            coef, c, r, fl = cheb_relu(a, b, k); flops += fl
            y = u[i].copy(); y[0] -= c; y /= r; flops += ma
            acc = coef[k] * y; acc[0] += coef[k - 1]; flops += 2 * ma      # Horner: k-1 truncated products
            for jj in range(k - 2, -1, -1):
                acc, f = A.mul(acc, y, active=t + 1); flops += f
                acc[0] += coef[jj]
            newp[i] = acc; kinds.append("cheb")
        p = newp
        P.append(p.copy())
        log.append(kinds)
    return P, flops, log


def endpoint_cost(A2):
    """Itemised cheapest sound endpoint: conditional second moments of all 90 margins on the 10 regions
    R_j = {x_j is 2nd largest} via model-free Gram tables G_j (M x M): L G_j L^T per j (L = 10 logit coefficient
    rows). FLOPs: 10 regions x (2 M^2 x 10 + 2 M x 100) plus first moments 10 x 2 M x 10."""
    M = A2.M
    return 10 * (2 * M * M * 10 + 2 * M * 100) + 10 * 2 * M * 10


def diag(model, A, P, rng, n=2_000_000, chunk=50_000, steps=True):
    """DIAGNOSTIC MC: surrogate accuracy s, per-step relative L2 error, conditional-moment Cantelli width."""
    Wi, Wh, Wo = model
    L = Wo @ P[-1]                                  # (10, M) logit coefficients
    ok_s = ok_t = 0
    e2 = np.zeros(T); h2 = np.zeros(T)
    # Cantelli pieces on R_j: sums of m and m^2 for margins m_jk restricted to {x_j 2nd}
    S1 = np.zeros((T, T)); S2 = np.zeros((T, T)); cnt = np.zeros(T)
    for _ in range(n // chunk):
        x = rng.standard_normal((chunk, T))
        B = A.basis(x)
        ls = B @ L.T
        lt = arc_eval.forward(model, x)
        tgt = np.argsort(x, 1)[:, -2]
        ok_s += (ls.argmax(1) == tgt).sum(); ok_t += (lt.argmax(1) == tgt).sum()
        if steps:
            h = np.zeros((chunk, H))
            for t in range(T):
                h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
                ph = B @ P[t].T
                e2[t] += ((ph - h) ** 2).sum(); h2[t] += (h ** 2).sum()
        for j in range(T):
            sel = tgt == j
            m = ls[sel, j:j + 1] - ls[sel]                  # m_jk, column j is 0
            S1[j] += m.sum(0); S2[j] += (m ** 2).sum(0); cnt[j] += sel.sum()
    # Cantelli union bound (uses only the surrogate's conditional moments, i.e. ignores propagation remainders):
    # P(wrong | R_j) <= sum_k min(1, var/(var + mean^2)) for mean > 0, else 1.
    mean = S1 / cnt[:, None]; var = S2 / cnt[:, None] - mean ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        cb = np.where(mean > 0, var / (var + mean ** 2), 1.0)
    for j in range(T):
        cb[j, j] = 0
    wrong_ub = (cnt / cnt.sum() * np.minimum(1, cb.sum(1))).sum()
    return dict(s=ok_s / n, mc_truth_check=ok_t / n, rel_l2=list(np.sqrt(e2 / h2)) if steps else None,
                cantelli_acc_lower=1 - wrong_ub)


def oracle_floor(model, d, rng, n_fit=4_000_000, n_eval=1_000_000, chunk=None):
    """DIAGNOSTIC D1: L2-optimal degree-d Hermite projection of the TRUE logits (MC coefficients) and its accuracy."""
    A = Algebra(T, d)
    chunk = chunk or max(5_000, int(2.5e7 // A.M))
    acc = np.zeros((A.M, T))
    done = 0
    while done < n_fit:
        x = rng.standard_normal((chunk, T))
        acc += A.basis(x).T @ arc_eval.forward(model, x); done += chunk
    C = (acc / done) / A.norm2[:, None]             # (M, 10)
    ok = 0; done = 0; se = 0.0; var = 0.0
    while done < n_eval:
        x = rng.standard_normal((chunk, T))
        lt = arc_eval.forward(model, x); lp = A.basis(x) @ C
        ok += (lp.argmax(1) == np.argsort(x, 1)[:, -2]).sum(); done += chunk
        se += ((lp - lt) ** 2).sum(); var += ((lt - lt.mean(0)) ** 2).sum()
    return dict(d=d, acc=ok / done, rel_l2_logits=float(np.sqrt(se / var)))


class PolyCoeffEstimator:
    """est(model, budget, rng) -> (estimate, evals). The deductive part is the propagation; the endpoint integral
    for the point estimate is NOT executable within budget (G1), so `est` returns the surrogate's accuracy from the
    diagnostic MC and reports evals as the FPE-equivalent of the deductive propagation only. Used for the G2 stand-in
    numbers, never as a valid deductive estimate."""

    def __init__(self, d, k):
        self.A, self.k = Algebra(T, d), k

    def __call__(self, model, budget, rng):
        P, fl, _ = propagate(model, self.A, self.k)
        r = diag(model, self.A, P, rng, steps=False, n=1_000_000)
        return r["s"], fl / FPE


def main(configs):
    z, names, truths = load_panel()
    out = {}
    for d, k in configs:
        A = Algebra(T, d); A.product_table()
        for nm in SCREEN:
            i = names.index(nm)
            model = (z["Wi"][i].astype(float), z["Wh"][i].astype(float), z["Wo"][i].astype(float))
            P, fl, log = propagate(model, A, k)
            ncheb = sum(kk.count("cheb") for kk in log)
            r = diag(model, A, P, np.random.default_rng(29100 + i), n=2_000_000 if d <= 3 else 500_000)
            p = truths[nm]["acc"]
            r.update(name=nm, d=d, k=k, truth=p, C_prop_fpe=fl / FPE, C_end_fpe=endpoint_cost(A) / FPE, n_cheb=ncheb)
            out[f"{nm}_d{d}k{k}"] = r
            print(json.dumps({kk: (round(v, 5) if isinstance(v, float) else v) for kk, v in r.items()
                              if kk != "rel_l2"}), "rel_l2:", np.round(r["rel_l2"], 3).tolist(), flush=True)
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    cfg = [(int(a[i]), int(a[i + 1])) for i in range(0, len(a), 2)] or [(3, 8)]
    res = main(cfg)
    tag = "_".join(f"d{d}k{k}" for d, k in cfg)
    json.dump(res, open(os.path.join(HERE, f"results_{tag}.json"), "w"), indent=1)
