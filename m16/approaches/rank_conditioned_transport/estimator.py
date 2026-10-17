"""Experimental H2 estimator interface; advisory budget, uncertified grid error."""
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import estimate


def est(model, budget, rng):
    """Return estimate and charged, hardware-dependent forward equivalents.

    Uses fixed hidden17/target8-per-half settings. Budget and rng do not change
    allocation. The caller must use the returned actual cost, not budget, when
    comparing methods. Calibration uses deterministic zero inputs only for
    timing; those outputs do not enter the accuracy estimate. Imports excluded.
    """
    wi, wh, wo = (np.asarray(w, dtype=float) for w in model)
    value, stats = estimate((wi, wh, wo), hidden_order=17, target_order=8,
                            mesh_power=2., target_rule='split_sign',
                            transition='polynomial_piece_cached', terminal_reuse=True)
    count = 32768
    start = time.process_time()
    inputs = np.zeros((count, len(wo)))
    hidden = np.zeros((count, len(wi)))
    for t in range(len(wo)):
        hidden = np.maximum(hidden @ wh.T+inputs[:, t, None]*wi, 0)
    np.argmax(hidden @ wo.T, axis=1)
    elapsed = time.process_time()-start
    if elapsed <= 0:
        raise ArithmeticError('forward timing resolution insufficient')
    return value, (stats['cpu_seconds']+elapsed)/(elapsed/count)
