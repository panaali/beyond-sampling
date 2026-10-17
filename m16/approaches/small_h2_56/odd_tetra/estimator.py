"""T71 fixed GL24 deductive estimator, with conservative counted diagnostics."""
import time
import numpy as np
try:
    from .counted import Audit, enumerate_correct, volume
except ImportError:
    from counted import Audit, enumerate_correct, volume

COST_REVIEWED = False  # DRI may enable after independent code/accounting review.
CLASSIFICATION = 'deductive geometry and analytic tetrahedron moments'


def calculate(model):
    start = time.process_time()
    audit = Audit()
    model = tuple(np.asarray(x, dtype=float) for x in model)
    model = (model[0].ravel(), model[1], model[2])
    audit.add('provided_model_validation', 128)
    if not all(np.isfinite(x).all() for x in model):
        raise ValueError('finite weights required')
    try:
        rays, geometry, _ = enumerate_correct(model, audit)
        geometry_cpu = time.process_time()-start
        estimate, integration = volume(rays, audit, order=24, inner_budget=1e-9)
    except Exception as exc:
        exc.partial_stats = dict(cost=audit.report(), cpu_seconds=time.process_time()-start)
        raise
    return estimate, dict(geometry=geometry, geometry_cpu_seconds=geometry_cpu,
                          correct_simplices=len(rays), integration=integration,
                          cost=audit.report(), cpu_seconds=time.process_time()-start,
                          affine_only_forward_equivalents=audit.known/65,
                          model_inputs_evaluated=0, cost_independently_reviewed=COST_REVIEWED,
                          geometry_and_roundoff_certified=False)


def est(model, budget, rng):
    """Fixed design; budget advisory, rng unused. No random model inputs.

    Until independent cost review, the harness receives infinite cost while
    calculate() exposes the full proposed scalar bound and component inventory.
    """
    value, stats = calculate(model)
    return value, stats['affine_only_forward_equivalents'] if COST_REVIEWED else float('inf')
