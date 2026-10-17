"""Fixed Sobol tie-face quadrature using a genuinely incremental exact walker.

API is the shared arc_eval / leaderboard protocol: est(model, budget, rng).
This is QUADRATURE, not a mechanistic solution. rng is deliberately ignored.
Pilot observations are reused and all model-dependent work is charged.
"""
import importlib.util
from pathlib import Path
import numpy as np

_spec = importlib.util.spec_from_file_location('_sci_tao_v2_core', Path(__file__).with_name('core.py'))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

DEDUCTIVE = False  # do not confuse deterministic numerical quadrature with mechanism
CLASS = 'quadrature'  # advisory; shared leaderboard class still requires DRI registration
FIXED_COST = False
ADVICE = None
# API is generic for all H>=1, T>=2. Current shared loader accepts only a list,
# so explicitly declare the normal zoo grid plus the random validation sizes.
SUPPORTS = [(H, T) for H in (1, 2, 3, 4, 5, 8, 12, 16, 32, 64) for T in range(2, 17)]
# Analytic proxy for the 160001-point reusable model-free Gaussian face tables.
# Fixed Sobol generation is model-free too and scales with requested nodes.
PRECOMPUTE_FLOPS = 30_000_000
RULE_SEED = 2026


def _run(model, kind, U):
    T = model[2].shape[0]
    masses, costs = [], []
    for i in range(0, len(U), 4096):
        y, a, b, tau = C.face_points(T, kind, U[i:i + 4096])
        mass, cost, _ = C.walk_mass(model, y, a, b, tau, kind)
        masses.append(mass)
        costs.append(cost)
    return np.concatenate(masses), np.concatenate(costs)


def est(model, budget, rng=None):
    estimate, cost, _ = estimate_details(model, budget)
    return estimate, cost


def estimate_details(model, budget):
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError('budget must be finite and positive')
    model = tuple(np.asarray(w, dtype=np.float64) for w in model)
    H, T = len(model[0]), model[2].shape[0]
    unit = C.fpe_flops(H, T)
    _, Z23, Z12 = C.tables(T)
    kinds = [('23', Z23), ('12', Z12)] if T >= 3 else [('12', Z12)]
    data = {}
    used = 0.
    # Same predeclared pilot heuristic as sci_tao, not fitted to any model.
    n0 = max(16, int(.25 * budget / 8 / len(kinds)))
    for i, (kind, _) in enumerate(kinds):
        F, cost = _run(model, kind, C.sobol(T, n0, RULE_SEED + i))
        data[kind] = [F, cost]
        used += cost.sum()
    # Counts reductions, std/mean, sqrt, and split arithmetic conservatively.
    allocation_flops = sum(8 * len(data[k][0]) + 24 for k, _ in kinds)
    used += allocation_flops / unit
    remaining = budget - used
    if remaining > 0:
        scores = {k: Z * data[k][0].std() / np.sqrt(data[k][1].mean()) for k, Z in kinds}
        total = sum(scores.values())
        for i, (kind, _) in enumerate(kinds):
            share = remaining * scores[kind] / total if total > 0 else remaining / len(kinds)
            add = int(share / data[kind][1].mean())
            if add > 0:
                have = len(data[kind][0])
                U = C.sobol(T, have + add, RULE_SEED + i)[have:]
                F, cost = _run(model, kind, U)
                data[kind][0] = np.r_[data[kind][0], F]
                data[kind][1] = np.r_[data[kind][1], cost]
                used += cost.sum()
    aggregate_flops = sum(2 * len(data[k][0]) + 4 for k, _ in kinds) + 1
    used += aggregate_flops / unit
    error = sum(Z * data[k][0].mean() for k, Z in kinds)
    info = {'faces': {k: len(data[k][0]) for k, _ in kinds},
            'walk_fpe': float(sum(data[k][1].sum() for k, _ in kinds)),
            'allocation_flops': allocation_flops, 'aggregation_flops': aggregate_flops,
            'cdf_flops_per_call_proxy': C.CDF_FLOPS}
    return float(1 - error), float(used), info
