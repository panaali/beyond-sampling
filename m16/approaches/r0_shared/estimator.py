"""T300 frozen estimator for HP2 (lenient bar, retrains): R0 point closure + one-anchor Newton boundary on the 2nd-vs-3rd
gap per position triple + model-free exact masses. Frozen 2026-10-18 = r0_integral/tb.py variant
(n23=1, newton=1, nblind=0, use12=False); no constant was fitted to any model. Charged cost ~1,752 FPE
(T196 FLOPs, multiply-add = 2, value + tangent = 2 forwards, Gauss-Legendre and Phi evaluations charged).

Step labels (Hilton):
  * R0 closure (7 low inputs at their conditional mean)      : declared structural approximation (deductive, approx.)
  * position triples and exact top-3 order-statistic masses : deductive, model-free
  * margin + exact tangent at one constructed anchor/triple  : constructed inputs, deductive along that line
  * extending that threshold to all (v2, g12) of the triple  : INDUCTIVE (assumes the rest behaves like the anchor)
  * 1st-vs-2nd boundary ignored, blind region counted wrong  : declared approximations
  * anchors / median gaps from a seeded model-free order-statistic table (no model calls)
The budget argument is ignored (fixed cost).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r0_integral"))
import r0  # noqa: E402
import tb  # noqa: E402

DEDUCTIVE = False          # one inductive step (threshold extrapolation within a triple); see docstring
VARIANT = dict(n23=1, newton=1, nblind=0, use12=False)


def est(model, budget=None):
    """model = (Wi (16,), Wh (16,16), Wo (10,16)) float arrays -> (accuracy estimate, charged FLOPs)."""
    import numpy as np
    m = tuple(np.asarray(w, dtype=np.float64) for w in model)
    return tb.est(m, **VARIANT)
