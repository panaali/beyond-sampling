"""Incremental tie-segment integration, with observed operation counters.

All cached hidden states are affine coefficients in the *absolute* line coordinate.
A gate switch invalidates only its suffix. Cached prefix roots remain candidates.
Geometry/node construction is the unchanged model-free sci_tao implementation.
"""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.special import ndtr

_spec = importlib.util.spec_from_file_location('_sci_tao_geometry_v2', Path(__file__).parents[1] / 'sci_tao/core.py')
geometry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(geometry)
tables, face_points, sobol, forward = geometry.tables, geometry.face_points, geometry.sobol, geometry.forward
SQ2 = np.sqrt(2.)
CDF_FLOPS = 50  # explicit analytic proxy, not a measured scipy instruction count


def fpe_flops(H, T):
    return 6000. * (T * (2 * H * H + 2 * H) + 2 * H * T) / 5760.


def walk_mass(model, y, a, b, tau, kind, eps=1e-10, max_cells=2000, details=False):
    """Return wrong Gaussian mass, actual-work FPE proxy, and cell count per line.

    `details=True` also returns per-line operation categories. A cell-limit failure
    raises instead of silently returning a truncated integral. T >= 2, any H >= 1.
    eps chooses right-side gates; this is numerical exact cell integration, not
    an interval-arithmetic certificate. Input lines must satisfy the tie geometry.
    """
    Wi, Wh, Wo = (np.asarray(w, dtype=np.float64) for w in model)
    y = np.asarray(y, dtype=np.float64)
    n, T = y.shape
    H = len(Wi)
    if T < 2 or Wh.shape != (H, H) or Wo.shape != (T, H) or kind not in ('12', '23'):
        raise ValueError('Expected bias-free ReLU (Wi[H], Wh[H,H], Wo[T,H]), T>=2, kind 12/23')
    a, b, tau = np.asarray(a), np.asarray(b), np.asarray(tau)
    if np.any(tau < 0) or not np.isfinite(tau).all() or not np.isfinite(y).all():
        raise ValueError('Finite nonnegative windows and finite face points required')
    d = np.zeros_like(y)
    rows = np.arange(n)
    d[rows, a] = 1 / SQ2
    d[rows, b] = -1 / SQ2
    first = np.minimum(a, b)
    target_neg, target_pos = (b, a) if kind == '23' else (a, b)
    # B + t*S is the exact hidden state within the cached activation cell.
    B = np.zeros((n, T + 1, H))
    S = np.zeros_like(B)
    roots = np.full((n, T), np.inf)
    at = -tau.copy()
    restart = first.copy()
    mass = np.zeros(n)
    cells = np.zeros(n, dtype=np.int64)
    names = ('prefix_steps', 'suffix_steps', 'readout_vectors', 'root_vectors',
             'envelope_sides', 'cdf_calls', 'recurrence_flops', 'root_flops',
             'readout_flops', 'envelope_flops', 'bookkeeping_flops')
    work = {name: np.zeros(n, dtype=np.int64) for name in names}
    active = np.flatnonzero(at < tau)
    # Constant prefix: one value recurrence only; no zero tangent matrix product.
    for s in range(T):
        ids = active[first[active] > s]
        if not len(ids):
            continue
        z = B[ids, s] @ Wh.T + y[ids, s, None] * Wi
        B[ids, s + 1] = np.maximum(z, 0.)
        work['prefix_steps'][ids] += 1
        work['recurrence_flops'][ids] += 2 * H * H + 2 * H
    for _ in range(max_cells):
        if not len(active):
            break
        tc = at[active] + eps * (1 + np.abs(at[active]))
        # Do not leap beyond a positive but tiny terminal interval.
        tc = np.minimum(tc, np.nextafter(tau[active], at[active]))
        probe = np.zeros(n)
        probe[active] = tc
        for s in range(T):
            ids = active[restart[active] <= s]
            if not len(ids):
                continue
            z0 = B[ids, s] @ Wh.T + y[ids, s, None] * Wi
            z1 = S[ids, s] @ Wh.T + d[ids, s, None] * Wi
            gates = z0 + probe[ids, None] * z1 > 0
            B[ids, s + 1] = np.where(gates, z0, 0.)
            S[ids, s + 1] = np.where(gates, z1, 0.)
            with np.errstate(divide='ignore', invalid='ignore'):
                r = -z0 / z1
            valid = (r > probe[ids, None]) & np.isfinite(r)
            roots[ids, s] = np.where(valid, r, np.inf).min(axis=1)
            work['suffix_steps'][ids] += 1
            work['recurrence_flops'][ids] += 4 * H * H + 9 * H
            work['root_vectors'][ids] += 1
            work['root_flops'][ids] += 7 * H - 1
        # Include future roots from cached prefix: discarding these misses cells.
        rr = roots[active]
        rr = np.where(rr > tc[:, None], rr, np.inf)
        rstep = rr.argmin(axis=1)
        nxt = np.minimum(tau[active], rr[np.arange(len(active)), rstep])
        if np.any(nxt <= at[active]):
            raise ArithmeticError('Segment walker failed to advance')
        L0 = B[active, T] @ Wo.T
        L1 = S[active, T] @ Wo.T
        work['readout_vectors'][active] += 2
        work['readout_flops'][active] += 2 * T * (2 * H - 1)
        work['bookkeeping_flops'][active] += 6 * T + 12
        cells[active] += 1
        lo, hi = at[active], nxt
        for side, target in ((0, target_neg), (1, target_pos)):
            p_lo = lo if side == 0 else np.maximum(lo, 0.)
            p_hi = np.minimum(hi, 0.) if side == 0 else hi
            valid = p_hi > p_lo
            # Work only on nonempty half-intervals.
            ids = active[valid]
            if not len(ids):
                continue
            local = np.flatnonzero(valid)
            k = target[ids]
            c = L0[local, k, None] - L0[valid]
            e = L1[local, k, None] - L1[valid]
            with np.errstate(divide='ignore', invalid='ignore'):
                r = -c / e
            lower = np.where(e > 0, r, -np.inf)
            upper = np.where(e < 0, r, np.inf)
            # Constant equal logits obey numpy's first-index tie rule exactly.
            bad = (e == 0) & ((c < 0) | ((c == 0) & (np.arange(T)[None] < k[:, None])))
            ii = np.arange(len(ids))
            lower[ii, k], upper[ii, k] = -np.inf, np.inf
            clo = np.maximum(lower.max(axis=1), p_lo[valid])
            chi = np.minimum(upper.min(axis=1), p_hi[valid])
            correct = np.where((chi > clo) & ~bad.any(axis=1), ndtr(chi) - ndtr(clo), 0.)
            mass[ids] += ndtr(p_hi[valid]) - ndtr(p_lo[valid]) - correct
            work['envelope_sides'][ids] += 1
            work['cdf_calls'][ids] += 4
            work['envelope_flops'][ids] += 20 * T + 12 + 4 * CDF_FLOPS
        at[active] = nxt
        restart[active] = rstep
        active = active[nxt < tau[active]]
    if len(active):
        raise RuntimeError(f'max_cells={max_cells} exhausted on {len(active)} segments')
    flop_keys = [k for k in work if k.endswith('_flops')]
    flops = sum(work[k] for k in flop_keys)
    result = (mass, flops / fpe_flops(H, T), cells)
    return result + (work,) if details else result
