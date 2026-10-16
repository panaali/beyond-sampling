"""Leaderboard interface for the probabilistic branch-and-bound (DEDUCTIVE: no random model inputs).
budget/cost in forward-pass equivalents (6e3 FLOPs). Estimate = certified mass L + undecided cells whose
representative point is classified correct (deterministic cell-centre rule). With leaderboard budgets
(<= 256k passes = 1.5e9 FLOPs) the B&B never certifies anything, so this is essentially a coarse
deterministic cubature. If the budget is below the cost of bounding the 180 root cells (~48k passes), only the
180 root centres are evaluated."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bb2  # noqa: E402
import cone  # noqa: E402
from core import FLOP_FWD, FLOP_BWD, FLOP_CROWNZ  # noqa: E402

DEDUCTIVE = True
_CELL = (FLOP_FWD + FLOP_BWD + FLOP_CROWNZ) / 6000.0


def est(model, budget, rng=None):
    root = np.arange(len(cone.ROOTS)); wl, wu = cone.root_boxes()
    if budget < 3 * len(root) * _CELL:
        mass = bb2.meas(root, wl, wu)
        cc = bb2.center_eval(root, wl, wu, model)
        return float(mass[cc].sum() + 0.5 * (1 - mass.sum())), float(len(root))
    log, _ = bb2.run(model=model, max_flops=0.5 * budget * 6000, verbose=False, batch=512)
    r = log[-1]
    return r["point_center"], float(r["flops"] / 6000 + sum(x["center_evals"] for x in log))
