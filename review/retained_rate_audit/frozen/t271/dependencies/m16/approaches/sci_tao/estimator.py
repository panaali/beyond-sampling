"""Leaderboard entry: deterministic co-area (tie-face) rule with a deterministic two-stage allocation.

est(model, budget, rng) -> (estimate, cost). rng is ignored: the nodes are a fixed scrambled-Sobol rule
(seed fixed in advance, identical for every model), so no random model inputs. Cost = forward-pass
equivalents actually used by the exact segment walks (value + tangent, recomputed from the step whose gate
flipped; shared prefix counted once). Stage 1 spends 25% of the budget evenly on R23 and R12 faces; stage 2
allocates the rest by Neyman (sd * Z / sqrt(cost per face)) estimated from stage 1. All faces are reused.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core as C

DEDUCTIVE = True
FIXED_COST = False
RULE_SEED = 2026


def _run(model, kind, U):
    T = model[2].shape[0]
    Fs, cs = [], []
    for i in range(0, len(U), 4096):
        y, a, b, tau = C.face_points(T, kind, U[i:i + 4096])
        F, c, _ = C.walk_mass(model, y, a, b, tau, kind)
        Fs.append(F); cs.append(c)
    return np.concatenate(Fs), np.concatenate(cs)


def est(model, budget, rng=None):
    model = tuple(np.asarray(w, dtype=np.float64) for w in model)
    T = model[2].shape[0]
    _, Z23, Z12 = C.tables(T)
    kinds = [("23", Z23), ("12", Z12)] if T >= 3 else [("12", Z12)]
    nmax = int(budget) + 64
    Us = {k: C.sobol(T, nmax, RULE_SEED + i) for i, (k, _) in enumerate(kinds)}
    data = {}
    n0 = max(16, int(0.25 * budget / 8 / len(kinds)))
    used = 0.0
    for k, _ in kinds:
        F, c = _run(model, k, Us[k][:n0]); data[k] = [F, c]; used += c.sum()
    rem = budget - used
    if rem > 0:
        sc = {k: Z * data[k][0].std() / np.sqrt(data[k][1].mean()) for k, Z in kinds}
        tot = sum(sc.values()) or 1.0
        for k, Z in kinds:
            share = rem * sc[k] / tot if sum(sc.values()) > 0 else rem / len(kinds)
            nadd = int(share / data[k][1].mean())
            if nadd > 0:
                n_have = len(data[k][0])
                F, c = _run(model, k, Us[k][n_have:n_have + nadd])
                data[k][0] = np.r_[data[k][0], F]; data[k][1] = np.r_[data[k][1], c]; used += c.sum()
    err = sum(Z * data[k][0].mean() for k, Z in kinds)
    return float(1 - err), float(used)
