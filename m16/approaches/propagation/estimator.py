"""Leaderboard entry: Gaussian-sum distribution propagation (gsf2.py). Deterministic; no model inputs sampled.
cost = counted FLOPs / 6e3 (forward-pass equivalents). The budget sets the branch-pruning threshold eps."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from gsf2 import gsf2

DEDUCTIVE = True


def est(model, budget, rng=None):
    # measured on the original model: eps=1e-2 -> ~?k, 1e-3 -> ~22k, 1e-4 -> ~117k passes (K=1)
    eps = float(np.clip(1e-3 * 22_000 / max(budget, 1), 1e-4, 3e-2))
    r = gsf2(tuple(np.asarray(w, float) for w in model), K=1, eps=eps)
    return r['acc'], r['flops'] / 6e3
