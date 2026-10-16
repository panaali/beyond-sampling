"""Leaderboard entry for sci_laplace: Laplace-tube quadrature (lqt.py). Deterministic: fixed point sets
(seed 0), no random model inputs, no model-fitted tables. Budget/cost in forward-pass equivalents
(tube's accounting: a linear cell = value+tangent recomputed from the flipped step, 2(T-t)/T)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lqt import e23, e12, eblind

K0 = 0.04
SHARE = (0.66, 0.27, 0.05)          # R23 faces / R12 faces / blind points (fixed; from the original model)
COST = (7.5, 8.0)                   # mean passes per face (original model), used only to size the rules


def est_seeded(model, budget, seed=0, parts=False):
    m23 = max(1, round(SHARE[0] * budget / (COST[0] * 360)))
    m12 = max(1, round(SHARE[1] * budget / (COST[1] * 45)))
    nb = max(11, int(SHARE[2] * budget))
    a, ca, _ = e23(model, m23, seed, K0)
    b, cb, _ = e12(model, m12, seed, K0)
    c, cc = eblind(model, nb, seed)
    acc = 1 - a - b - c
    if parts:
        return acc, ca + cb + cc, dict(e23=a, e12=b, eb=c)
    return float(acc), float(ca + cb + cc)


def est(model, budget, rng=None):
    return est_seeded(model, budget, 0)


DEDUCTIVE = True
FIXED_COST = False
