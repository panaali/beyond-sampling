"""HP1 entry points (T274a, frozen; v5.py unchanged since 11238bc3 / f7e14e83).
est(model, budget) -> (estimate, cost_flops); model = (Wi (16,), Wh (16,16), Wo (10,16)) float64; budget in FPE.
Default = probe_v5q (the method that passed LB3 under the amended rule); est_v5a is the probe-proper variant.
Cost: 1 network forward per node = 6000 FLOPs (LB3 FPE, T196 convention). No assumption about the accuracy regime:
the estimate is the fraction of exact-law nodes on which the network returns the rank-2 position."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5

FPE_FLOPS = 6000.0


def est(model, budget):
    e, c = v5.est_full(model, budget)
    return e, c * FPE_FLOPS


def est_v5a(model, budget):
    e, c = v5.est(model, budget)
    return e, c * FPE_FLOPS
