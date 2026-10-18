"""T317e frozen estimator: T317d (estimator_d.py) as one self-contained file (cont_e.py) with no Monte Carlo anywhere;
estimates and FLOPs bitwise identical to T317d. G = 20. Deductive. est(model, budget) -> (estimate, FLOPs as executed)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cont_e  # noqa: E402

DEDUCTIVE = True


def est(model, budget=None):
    return cont_e.est(model)
