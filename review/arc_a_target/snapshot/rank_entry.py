"""Leaderboard entry for the rank-ordering estimator (see README.md). Deterministic; rng unused.
Minimum cost is about 91k forward-pass equivalents (7,920 classes x frozen-gate linear map + quadrature);
budgets below that still return the cheapest version (Q=16), whose cost then exceeds the budget."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_est import est_rank  # noqa: E402

DEDUCTIVE = True


def est(model, budget, rng):
    Q = 16 if budget < 104_000 else (64 if budget < 361_000 else 1024)
    return est_rank(model, Q=Q)
