"""T286 second frozen entry, selected AFTER the DEV2 run (post hoc, on the open DEV2 retrains; the pre-registered
candidate is hp2_entry.est). Interface est(model, budget) -> (estimate, cost_flops). Deterministic, fixed cost.
FROZEN 2026-10-18 at the commit that adds this file.

  est  probe series m16/estimate.py v1 (2~3 flip-gap bisection at canonical inputs, model-free gap law, blind error
       8/9), bisection iters 8, all 720 triples. All steps deductive (blind constant argued, not computed).
       ~3.9k FPE = 2.3e7 FLOPs (1 forward pass per probe). DEV2 F_max 76.5 (fused 76.5, torch 72.5); LB3 seed F 77.6,
       LB3 F_max 1042 (heavy-noise copies: rank-4+ and 1~2 errors are not modelled)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cf import FPE  # noqa: E402
from probe import est_probe  # noqa: E402

DEDUCTIVE = True


def est(model, budget=None):
    e, c = est_probe(model, version=1, iters=8, stride=1)
    return e, c * FPE
