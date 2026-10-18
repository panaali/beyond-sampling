"""T306 frozen estimator = R1W V1 (the pre-registered selection winner among V1 / A / B / A+B on DEV2; A and B did
not help). Deterministic. est(model, budget) -> (accuracy estimate, charged FLOPs); budget ignored (fixed cost
~13.7k FPE = ~7.9e7 FLOPs, T196 convention as implemented).

Steps (Hilton labels): R0-family closure with v4 at the triple's latest free position (declared structural
approximation); v4's value at its conditional mean given v3, a 1-node Gauss rule on its exact law (numerical
integration, declared); exact segment walks of the piecewise-linear margin along the g23 and g12 lines at
conditional-mean transverse points, walk length .5 (deductive along the line; transverse point and walk length
declared); blind-region g12 assumed correct (declared); exact model-free order-statistic masses (deductive)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r1w2  # noqa: E402

DEDUCTIVE = True


def est(model, budget=None):
    return r1w2.est(model)
