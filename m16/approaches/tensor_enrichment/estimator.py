"""Unvalidated fixed weighted residual recipe with partial FPE accounting.

This adapter satisfies the shared signature; its returned cost is explicitly a
partial operation proxy, not a reviewed portable cost or a budget guarantee.
"""
from .enrich import CachedOracle, Enriched, cross

CLASS = 'unreviewed'
DEDUCTIVE = True
FIXED_COST = True
COST_REVIEWED = False
COST_IS_PARTIAL_PROXY = True
SUPPORTS = [(2, 4), (4, 4)]
ADVICE = None


def estimate_details(model):
    cross.WALK = cross.C.walk_mass
    _, z23, z12 = cross.C.tables(model[2].shape[0])
    unit = cross.C.fpe_flops(len(model[0]), model[2].shape[0])
    error, cost, records = 0., 0., []
    for kind, factor in (('23', z23), ('12', z12)):
        grid = cross.FaceGrid(model, kind, 3)
        cache = CachedOracle(grid)
        fit = Enriched(cache, grid.shape, grid.weights, weighted=True, rounds=2)
        integral = fit.integral(); report = fit.report()
        algebra = report['original_cross_algebra_proxy']+sum(report['enrichment_work'].values())
        error += factor*integral
        cost += grid.cost+algebra/unit
        records.append(dict(kind=kind, integral=integral, report=report,
                            walker_fpe_proxy=grid.cost, partial_algebra_proxy=algebra))
    return float(1-error), float(cost), records


def est(model, budget, rng=None):
    estimate, partial_cost, _ = estimate_details(model)
    return estimate, partial_cost
