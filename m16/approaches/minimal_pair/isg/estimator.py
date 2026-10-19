"""T391 entry (minimal_pair/isg): T10's frozen recipe with ONE global scrambled-Sobol rule per sign stratum, whose
configuration coordinate U[:,0] is mapped through an importance density over tie-position cells (a, b):
    q_cell ~ w_cell * rel_sd(stratum, cell) / sqrt(cost(stratum, cell))   (T389's frozen DEV2 table, ../pos/table.npz)
Each face gets the weight w_cell / q_cell, so E[weighted F] is unchanged. All other coordinates are as in the frozen
recipe. Stratum totals: frozen pilot (power-of-2 prefixes of the same rule, incremental) + Neyman, all points reused.
Deterministic (rng ignored, fixed nodes). Cost as executed (walk.py). No weight-dependent input: the table is a DEV2
population prior (T389 D11)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import core as C  # noqa: E402
import estimator as E0  # noqa: E402
import hom  # noqa: E402
import walk as W  # noqa: E402

DEDUCTIVE = True
FIXED_COST = False
SUPPORTS = [(16, 10)]
TAB = np.load(os.path.join(os.path.dirname(HERE), "pos", "table.npz"))
PRECOMPUTE_FLOPS = 0
RULE_SEED = 14_142
PILOT = 0.25


def _importance(T, kind, j):
    """Per-configuration sampling probabilities and weights w/q (configs of one cell share q equally)."""
    cfg = C.configs(T, kind)
    cell = cfg[:, 0] * T + cfg[:, 1]
    w_cfg = np.full(len(cfg), 1.0 / len(cfg))
    s = TAB["rel_sd"][j][cell] / np.sqrt(TAB["cost"][j][cell])
    q_cfg = w_cfg * np.maximum(s, 1e-3 * s.max()); q_cfg /= q_cfg.sum()
    return np.r_[0, np.cumsum(q_cfg)], w_cfg / q_cfg


def _run(model, T, kind, h, j, U, chunk=4096):
    edges, ratio = _importance(T, kind, j)
    ncfg = len(ratio)
    ci = np.clip(np.searchsorted(edges, U[:, 0], side="right") - 1, 0, ncfg - 1)
    U = U.copy(); U[:, 0] = (ci + 0.5) / ncfg            # face_points_strat picks config floor(U0 * ncfg) = ci
    vals, costs = [], []
    for i in range(0, len(U), chunk):
        y, a, b, tau, wt = E0.face_points_strat(T, kind, h, U[i:i + chunk])
        r = np.linalg.norm(y, axis=1)
        F, c, _ = W.walk(model, y / r[:, None], a, b, np.minimum(tau / r, 1e8), kind, cdf=hom.tcdf(T))
        vals.append(wt * F * ratio[ci[i:i + chunk]]); costs.append(c)
    return np.concatenate(vals), np.concatenate(costs)


def est(model, budget, rng=None, rule_seed=RULE_SEED, pilot=PILOT):
    model = tuple(np.asarray(w, dtype=np.float64) for w in model)
    T = model[2].shape[0]
    S = E0.strata(T)
    nmax = 1 << int(np.ceil(np.log2(max(budget, 64) / 2 + 64)))
    rules = {j: C.sobol(T, nmax, rule_seed + 101 * j) for j in range(len(S))}
    M = {j: E0.stratum_mass(T, kind, h) for j, (kind, h) in enumerate(S)}
    data = {j: [np.zeros(0), np.zeros(0)] for j in range(len(S))}
    used = 0.0

    def add(j, n_new):
        nonlocal used
        kind, h = S[j]; n_have = len(data[j][0]); n_new = min(n_new, nmax - n_have)
        if n_new <= 0:
            return
        v, c = _run(model, T, kind, h, j, rules[j][n_have:n_have + n_new])
        data[j][0] = np.r_[data[j][0], v]; data[j][1] = np.r_[data[j][1], c]; used += c.sum()

    n0 = 16
    while True:
        for j in range(len(S)):
            add(j, n0 - len(data[j][0]))
        if used >= pilot * budget or 2 * n0 > nmax:
            break
        n0 *= 2
    rem = budget - used
    if rem > 0:
        sc = {j: M[j] * data[j][0].std() / np.sqrt(data[j][1].mean()) for j in data}
        tot = sum(sc.values())
        for j in data:
            add(j, int(rem * (sc[j] / tot if tot > 0 else 1 / len(data)) / data[j][1].mean()))
    return float(1 - sum(M[j] * data[j][0].mean() for j in data)), float(used)
