"""T317 frozen estimator: R1W V1 estimates (identical, max diff 6.7e-16 on 80 DEV2) computed with continuity restarts and
an executed B2 prefix tree; G = 48 (G = 16 failed the DRI's 1e-6 equivalence ruling). est(model, budget) ->
(accuracy estimate, FLOPs as executed, Directive 8 convention). Deterministic; budget ignored.
Labels: as R1W V1 (r1_walk/estimator.py): deductive steps + declared structural approximations, no inductive step."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cont  # noqa: E402

DEDUCTIVE = True


def est(model, budget=None):
    return cont.est(model, G=48)
