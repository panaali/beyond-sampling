"""Leaderboard interface: great-circle exact integration + orthogonal frames + model-free control variate.

est(model, budget, rng) -> (estimate, cost). Sampling-based (random planes), unbiased.
Cost accounting (forward-pass equivalents): each gate piece runs one RNN pass carrying TWO coefficient
vectors (cos and sin parts) = 2 passes, plus within-piece breakpoint solving and argmax checks at up to
181 sub-interval midpoints (~10 logits x 2 coeffs each, ~7e3 FLOPs) = ~1 pass. So cost = 3 * pieces.
Model-free work (stratum features on a 512-point grid per circle, CV means P) is not counted; the CV
coefficients BETA were fitted once on 2000 circles of the ORIGINAL model (pilot_0.npz) and are frozen.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", ".."))
from gcirc import walk, frame_planes  # noqa: E402
from stratified import stratum, N_STRATA  # noqa: E402

DEDUCTIVE = False
COST_PER_PIECE = 3.0
PIECES_GUESS = 250.0
P = np.array([0.01077555, 0.0361162, 0.0528672, 0.0842093, 0.1535832, 0.24731025, 0.4151383])
BETA = np.array([-0.8377010043228481, -0.4239106444771151, -0.0937009226960446, -0.046115169085689196,
                 -0.01078672448735653, 0.012180878739851214])
_G = 512
_TH = np.arange(_G) / _G * 2 * np.pi


def est(model, budget, rng):
    nf = max(1, int(budget // (5 * PIECES_GUESS * COST_PER_PIECE)))
    U, V = frame_planes(nf, rng)
    f, pieces, _ = walk(model, U, V)
    feat = np.stack([np.bincount(stratum(np.cos(_TH)[:, None] * U[i] + np.sin(_TH)[:, None] * V[i]),
                                 minlength=N_STRATA) / _G for i in range(len(f))])
    fcv = f - (feat[:, :-1] - P[:-1]) @ BETA
    return float(fcv.mean()), float(COST_PER_PIECE * pieces.sum())
