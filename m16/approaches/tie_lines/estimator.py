"""Leaderboard entry: joint (g12 x g23) model-free strata + tie-swap antithetics (joint.py; see README.md).
Sampling-based, unbiased.

est(model, budget, rng) -> (estimate, cost). Cost in forward-pass equivalents:
  model evaluations (1 each; a swapped copy is a 2nd evaluation)
  + input generation: every candidate Gaussian vector drawn (also the rejected ones) at GEN_COST each
    (10 normals + sort of 10 + gap arithmetic ~ 200 FLOPs vs ~6e3 for a forward pass).
Inputs for all 170 strata are drawn from ONE shared pool (stop when every stratum has its quota), so the number
of candidates is max_h n_h/P_h instead of sum_h n_h/P_h. Within each stratum the accepted draws are i.i.d.
from the stratum's conditional law, so the estimator stays unbiased.
Precomputation reused across models (declared, joint_design.json): stratum masses P from 4e7 model-free draws,
and the allocation from a pilot on the ORIGINAL model only (1e6 draws; not counted, as for stratified.py).
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../.."))
from joint import est_joint                                     # noqa: E402


DEDUCTIVE = False


def est(model, budget, rng):
    return est_joint(model, budget, rng, count_gen=True)
