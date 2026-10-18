"""T223 leaderboard entry: sci_tao's deterministic co-area tie-slicing on the non-blind set + a fixed Sobol rule on the
blind set B = {at most one positive input} (P(B) = (T+1)/2^T exactly).

    P(wrong) = Z23 E23[F_N] + Z12 E12[F_N] + P(B) * E_B[1 wrong]

(core_nb.py). Three strata with fixed, model-independent scrambled-Sobol rules (seeds fixed in advance): R23 faces,
R12 faces, blind points. Allocation exactly as sci_tao's estimator: stage 1 spends 25% of the budget evenly on the
strata (R23/R12 at sci_tao's guess of 8 FPE per face, blind points at 1 FPE), stage 2 allocates the rest by Neyman
(sd * weight / sqrt(cost per node)) from stage 1; all nodes reused. rng is ignored (deterministic; class quadrature).
Cost: face walks are charged at 2.23x their counted forward-pass equivalents (the DRI audit's measured factor for the
identical sci_tao walk code, OVERRIDES["sci_tao"]); each blind point is one forward pass = 1 FPE.
The blind set cannot be enumerated exactly: inside B the target's prefix regions still grow 12-19x per step (T218
blind sub-test), and the trivial bound [0, P(B)] = 1.07 pp is far wider than the sampling SE."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core_nb as N

C = N.C
DEDUCTIVE = True
FIXED_COST = False
RULE_SEED = 2026          # same face rules as sci_tao (seeds 2026, 2027); blind rule seed 2223
BLIND_SEED = 2223
WALK_FACTOR = 2.23


def _faces(model, kind, U):
    T = model[2].shape[0]; Fs, cs = [], []
    for i in range(0, len(U), 4096):
        y, a, b, tau = C.face_points(T, kind, U[i:i + 4096])
        F, c, _ = N.walk_mass_nb(model, y, a, b, tau, kind)
        Fs.append(F); cs.append(WALK_FACTOR * c)
    return np.concatenate(Fs), np.concatenate(cs)


def _blind(model, U):
    T = model[2].shape[0]; out = []
    for i in range(0, len(U), 65536):
        out.append(N.blind_wrong(model, N.blind_points(T, U[i:i + 65536])))
    w = np.concatenate(out); return w, np.ones_like(w)


def est(model, budget, rng=None):
    model = tuple(np.asarray(w, dtype=np.float64) for w in model)
    T = model[2].shape[0]
    _, Z23, Z12 = C.tables(T); PB = (T + 1) / 2.0 ** T
    strata = [("23", Z23, 8.0), ("12", Z12, 8.0), ("B", PB, 1.0)] if T >= 3 else [("12", Z12, 8.0), ("B", PB, 1.0)]
    nmax = int(budget) + 64
    U = {k: (N.sobol(T + 1, nmax, BLIND_SEED) if k == "B" else C.sobol(T, nmax, RULE_SEED + i)) for i, (k, _, _) in enumerate(strata)}
    run = lambda k, u: _blind(model, u) if k == "B" else _faces(model, k, u)
    data = {}; used = 0.0
    for k, _, guess in strata:
        n0 = max(16, int(0.25 * budget / guess / len(strata)))
        F, c = run(k, U[k][:n0]); data[k] = [F, c]; used += c.sum()
    rem = budget - used
    if rem > 0:
        sc = {k: Z * data[k][0].std() / np.sqrt(data[k][1].mean()) for k, Z, _ in strata}
        tot = sum(sc.values())
        for k, Z, _ in strata:
            share = rem * sc[k] / tot if tot > 0 else rem / len(strata)
            nadd = int(share / data[k][1].mean())
            if nadd > 0:
                n_have = len(data[k][0])
                F, c = run(k, U[k][n_have:n_have + nadd])
                data[k][0] = np.r_[data[k][0], F]; data[k][1] = np.r_[data[k][1], c]; used += c.sum()
    err = sum(Z * data[k][0].mean() for k, Z, _ in strata)
    return float(1 - err), float(used)
