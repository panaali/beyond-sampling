"""T317d frozen estimator (T322): T317c with every model-free table precomputed once at import (cont_d.TABLES, listed by
cont_d.table_sizes()) and exact-law transverse points instead of seeded Monte Carlo ones. G = 20.
Deductive (no random model inputs). est(model, budget) -> (estimate, FLOPs as executed)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cont_d  # noqa: E402

DEDUCTIVE = True


def est(model, budget=None):
    return cont_d.est(model)
