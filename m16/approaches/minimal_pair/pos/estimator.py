"""T389 entry (minimal_pair/pos): T10's frozen recipe with node allocation WITHIN each sign stratum over tie-position
cells (a, b), set before walking by a fixed table (pos/table.npz: relative sd and cost per (stratum, cell), fitted on
40 DEV2 retrains only, d10_cells.py / d10_cv.py). The weights-only feature tried (position-influence norms, d11) added
nothing (held-out 1.000x), so the predictor is model-independent. Stratum totals: the frozen pilot + Neyman. The pilot
points are post-stratified into the cells (exact cell weights from exchangeability). Deterministic; cost as executed."""
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
TAB = np.load(os.path.join(HERE, "table.npz"))
PRECOMPUTE_FLOPS = 0          # the table is model-free (fitted on DEV2), loaded, not charged
RULE_SEED = 16_180
PILOT = 0.25


def _walk(model, T, kind, h, U, cell=None):
    """Faces of stratum h; if cell = (a, b) is given, U[:,0] is re-mapped into that cell's configurations."""
    if cell is not None:
        cfg = C.configs(T, kind)
        idx = np.nonzero((cfg[:, 0] == cell[0]) & (cfg[:, 1] == cell[1]))[0]
        k = np.minimum((U[:, 0] * len(idx)).astype(int), len(idx) - 1)
        U = U.copy(); U[:, 0] = (idx[k] + 0.5) / len(cfg)
    y, a, b, tau, wt = E0.face_points_strat(T, kind, h, U)
    r = np.linalg.norm(y, axis=1)
    F, c, _ = W.walk(model, y / r[:, None], a, b, np.minimum(tau / r, 1e8), kind, cdf=hom.tcdf(T))
    return wt * F, c, a * T + b


def est(model, budget, rng=None, rule_seed=RULE_SEED, pilot=PILOT):
    model = tuple(np.asarray(w, dtype=np.float64) for w in model)
    T = model[2].shape[0]
    S = E0.strata(T)
    used = 0.0
    nmax = 1 << int(np.ceil(np.log2(max(budget, 64) / 2 + 64)))
    pil = {j: (np.zeros(0), np.zeros(0), np.zeros(0, int)) for j in range(len(S))}
    n0, have = 16, 0
    while True:                                   # frozen pilot: equal power-of-2 Sobol prefixes per stratum, incremental
        for j, (kind, h) in enumerate(S):
            U = C.sobol(T, nmax, rule_seed + 101 * j)[have:n0]
            F, c, ab = _walk(model, T, kind, h, U); used += c.sum()
            pil[j] = (np.r_[pil[j][0], F], np.r_[pil[j][1], c], np.r_[pil[j][2], ab])
        have = n0
        if used >= pilot * budget or 2 * n0 > nmax:
            break
        n0 *= 2
    M = {j: E0.stratum_mass(T, kind, h) for j, (kind, h) in enumerate(S)}
    cells = {}
    for j, (kind, h) in enumerate(S):
        cfg = C.configs(T, kind)
        ab = np.unique(cfg[:, 0] * T + cfg[:, 1])
        w = np.array([np.mean(cfg[:, 0] * T + cfg[:, 1] == x) for x in ab])
        cells[j] = (ab, w)
    rem = budget - used
    sc = {j: M[j] * pil[j][0].std() / np.sqrt(pil[j][1].mean()) for j in pil}
    tot = sum(sc.values()) or 1.0
    data = {}
    extra = 0.0
    for j, (kind, h) in enumerate(S):
        ab, w = cells[j]
        Fp, cp, abp = pil[j]
        nj = max(0, int(rem * sc[j] / tot / max(cp.mean(), 1e-3)))
        q = w * TAB["rel_sd"][j][ab] / np.sqrt(TAB["cost"][j][ab]); q = q / q.sum()
        nc = np.floor(nj * q).astype(int)
        for x, (cell, wc) in enumerate(zip(ab, w)):
            vals = list(Fp[abp == cell])
            if nc[x] > 0:
                U = C.sobol(T, max(nc[x], 2), rule_seed + 7919 * (j + 1) + 13 * int(cell))[:nc[x]]
                F, c, _ = _walk(model, T, kind, h, U, cell=(cell // T, cell % T)); extra += c.sum(); vals += list(F)
            data[(j, cell)] = (wc, np.mean(vals) if vals else 0.0)
    err = sum(M[j] * sum(wc * m for (jj, _), (wc, m) in data.items() if jj == j) for j in range(len(S)))
    return float(1 - err), float(used + extra)
