"""T317f frozen estimator (T335): T317e + the declared screen C1 on all lines (cont_f.py, self-contained, no Monte
Carlo anywhere). G = 20, tmax .5. Sampling-free at runtime; deductive structure plus 7 declared approximations
(README). est(model, budget) -> (estimate, FLOPs as executed)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cont_f  # noqa: E402

DEDUCTIVE = True


def est(model, budget=None):
    return cont_f.est(model)
