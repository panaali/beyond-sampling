"""Leaderboard interface for sci_feynman: deterministic tie-segment quadrature (see README.md).
est(model, budget, rng) -> (estimate, cost in forward-pass equivalents). rng is ignored (fixed rule)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feyn  # noqa: E402

DEDUCTIVE = True      # no random model inputs: one fixed, model-independent node set per budget
FIXED_COST = False


def est(model, budget, rng=None):
    acc, cost, _ = feyn.estimate(tuple(model), budget, design="sobol", shift_seed=None)
    return float(acc), float(cost)
