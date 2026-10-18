"""T286 frozen entries for sealed-panel scoring (T281 scorer). Interface est(model, budget) -> (estimate, cost_flops);
model = (Wi (16,), Wh (16,16), Wo (10,16)) float64. Deterministic (no random inputs; budget ignored: fixed cost).
FROZEN 2026-10-18 at the commit that adds this file; no constant changes after it.

  est      CANDIDATE (pre-registered rule: lowest LB3 F_max among all-deductive settings):
           rank_exact with Q=8 Sobol points, K pooled into 3 gate groups, all 720 top-3 triples.
           Steps: canonical gate points (model-free) -> exact frozen-gate linear maps (deductive approximation)
           -> exact integration of the linear surrogate over the model-free law of x | (ordering, K) (deductive).
           Cost ~25.8k FPE = 1.53e8 FLOPs. LB3 F_max 112 (seed group); it is not expected to pass either bar.
  est_ind  REFERENCE ONLY (has an inductive step: a fixed 1/16 subset of the position triples):
           rank_exact Q=4, 3 gate groups, stride 16. ~1.7k FPE = 1.0e7 FLOPs. LB3 F_max 16.7."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cf import FPE  # noqa: E402
from rx import est_rx  # noqa: E402

DEDUCTIVE = True


def est(model, budget=None):
    e, c = est_rx(model, Q=8, stride=1, kgroups=3)
    return e, c * FPE


def est_ind(model, budget=None):
    e, c = est_rx(model, Q=4, stride=16, kgroups=3)
    return e, c * FPE
