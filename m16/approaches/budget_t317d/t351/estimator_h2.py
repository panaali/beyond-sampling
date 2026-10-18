"""T353 frozen estimator (T317h'): T317f + T339 spread lows (s = -1) + T347 tilt correction of the line-set product,
crossing slopes from the walk's own cell state (cont_h2). Sampling-free at runtime (deductive + declared approximations).
est(model, budget=None) -> (estimate, FLOPs as executed under Directive 8)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cont_h2  # noqa: E402

DEDUCTIVE = True


def est(model, budget=None):
    return cont_h2.est(model)
