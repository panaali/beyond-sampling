"""T317c frozen estimator: T317b's estimates (max |diff| 0.0 on all 80 DEV2) with two exact, executed cost savings:
merging touching correct pieces before the Phi profile (telescoping), and model-free Phi tables at a = 0 / the tail.
correct_interval charged at 216 FLOPs. G = 20. est(model, budget) -> (estimate, FLOPs as executed)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cont_c  # noqa: E402

DEDUCTIVE = True


def est(model, budget=None):
    return cont_c.est(model, G=20)
