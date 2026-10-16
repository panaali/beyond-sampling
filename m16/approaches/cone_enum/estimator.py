"""Leaderboard entry for cone_enum: 1 - (Gaussian mass of error cones found by a seeded best-first facet
walk).  Cost = all model-dependent FLOPs (seed bisection passes, cone construction, LPs, hit-and-run +
importance-sampling integrator, facet-crossing checks) / 5760 FLOPs per forward pass.
No random model inputs: seeds are canonical rank-order points; the integrator's random numbers are
model-free (they sample inside an already-built polyhedral cone)."""
import sys
sys.path.insert(0, '/home/user/clean-neurons/m16')
sys.path.insert(0, '/home/user/clean-neurons/m16/approaches/cone_enum')
from walk import est_cone

DEDUCTIVE = True


def est(model, budget, rng=None):
    return est_cone(model, budget, rng)
