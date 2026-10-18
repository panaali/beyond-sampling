"""Leaderboard entry: stratified Rao-Blackwellised estimator (control_variate approach).

est(model, budget, rng) -> (estimate, cost). Cost in forward-pass equivalents: each prefix x0..x8 costs
COST_Q = 1.6 (9 RNN steps + exact piecewise-linear integration of the last input; see rb9.py).
Model-free precomputation reused across models (free, declared in README): precomp.npz (stratum
probabilities, 20M Gaussian draws) and alloc.json (allocation from a pilot on the ORIGINAL model).
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cv

DEDUCTIVE = False
_alloc = np.array(json.load(open(os.path.join(HERE, "alloc.json")))["plain"])
est = cv.make(strat=True, cv=None, alloc=_alloc)
