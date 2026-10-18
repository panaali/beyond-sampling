"""T317b frozen estimator: identical to T317 (estimator.py) except G = 20 Gauss-Legendre nodes, selected by the DRI's
pre-registered rule (smallest G in {20, 24, 32} with max |est(G) - est(48)| <= 1e-4 pp on all 80 DEV2: G=20 moves the
estimate by at most 1.4e-6 pp). est(model, budget) -> (estimate, FLOPs as executed). Deterministic; budget ignored."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cont  # noqa: E402

DEDUCTIVE = True


def est(model, budget=None):
    return cont.est(model, G=20)
