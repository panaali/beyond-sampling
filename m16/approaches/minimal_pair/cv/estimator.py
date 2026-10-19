"""T380 entry (minimal_pair/cv): T10's frozen recipe with each sign stratum split along the segment into a NEAR part
(|s| < DELTA, s = radial-normalised offset from the tie) and a FAR part (DELTA <= |s| < tau). The two parts are integrated
on independent fixed Sobol rules with their own Neyman allocation: far-from-tie cells are expensive and rarely
misclassified, so they get fewer nodes. Form and DELTA chosen on leaderboard DEV models at T=4/T=5 only (README, D6/D7).
Deterministic: rng ignored, fixed node sets. Cost as executed (walk.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core as C  # noqa: E402
import estimator as E0  # noqa: E402
import hom  # noqa: E402
import walk as W  # noqa: E402

DEDUCTIVE = True
FIXED_COST = False
SUPPORTS = [(16, T) for T in (3, 4, 5, 6, 8, 10)]
PRECOMPUTE_FLOPS = 0
RULE_SEED = 27_182
DELTA = 0.03
PILOT = 0.25


def _part(model, T, kind, h, U, part, chunk=4096):
    vals, costs = [], []
    cdf = hom.tcdf(T)
    for i in range(0, len(U), chunk):
        y, a, b, tau, wt = E0.face_points_strat(T, kind, h, U[i:i + chunk])
        r = np.linalg.norm(y, axis=1); w = y / r[:, None]; tw = np.minimum(tau / r, 1e8)
        if part == 0:
            F, c, _ = W.walk(model, w, a, b, np.minimum(tw, DELTA), kind, cdf=cdf)
        else:
            F = np.zeros(len(w)); c = np.zeros(len(w)); far = np.nonzero(tw > DELTA)[0]
            if len(far):
                for lo, hi in ((-tw[far], np.full(len(far), -DELTA)), (np.full(len(far), DELTA), tw[far])):
                    f, cc, _ = W.walk(model, w[far], a[far], b[far], hi, kind, cdf=cdf, lo=lo)
                    F[far] += f; c[far] += cc
        vals.append(wt * F); costs.append(c)
    return np.concatenate(vals), np.concatenate(costs)


def est(model, budget, rng=None, rule_seed=RULE_SEED, pilot=PILOT, details=False):
    model = tuple(np.asarray(w, dtype=np.float64) for w in model)
    T = model[2].shape[0]
    S = [(kind, h, part) for kind, h in E0.strata(T) for part in (0, 1)]
    nmax = 1 << int(np.ceil(np.log2(max(budget, 64) / 2 + 64)))
    rules = {}
    M = {j: E0.stratum_mass(T, kind, h) for j, (kind, h, _) in enumerate(S)}
    data = {j: [np.zeros(0), np.zeros(0)] for j in range(len(S))}
    used = 0.0

    def add(j, n_new):
        nonlocal used
        kind, h, part = S[j]; n_have = len(data[j][0])
        n_new = min(n_new, nmax - n_have)
        if n_new <= 0:
            return
        if j not in rules:
            rules[j] = C.sobol(T, nmax, rule_seed + 101 * j)
        v, c = _part(model, T, kind, h, rules[j][n_have:n_have + n_new], part)
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
        cm = {j: max(data[j][1].mean(), 1e-3) for j in data}
        sc = {j: M[j] * data[j][0].std() / np.sqrt(cm[j]) for j in data}
        tot = sum(sc.values())
        for j in data:
            share = rem * (sc[j] / tot if tot > 0 else 1 / len(data))
            add(j, int(share / cm[j]))
    err = sum(M[j] * data[j][0].mean() for j in data)
    if details:
        return float(1 - err), float(used), {S[j]: (M[j] * data[j][0].mean(), len(data[j][0])) for j in data}
    return float(1 - err), float(used)
