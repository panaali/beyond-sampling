"""Fixed target quantile, exact scalar terminal, numerical 2D transport.

The input-tail omission is bounded; interpolation and quadrature errors are not.
"""
from collections import defaultdict
from functools import lru_cache
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr, ndtri, roots_jacobi
from scipy.interpolate import RegularGridInterpolator


def normal_mass(lo, hi):
    if hi <= lo:
        return 0.
    return float(ndtr(-lo)-ndtr(-hi)) if lo >= 0 else float(ndtr(hi)-ndtr(lo))


def decision_intervals(model, hidden, target_value, label, limit, stats, target_last=None):
    """Correct scalar-input intervals; optional fixed target-last suffix."""
    wi, wh, wo = model
    pieces = [(-limit, limit, np.zeros(2), np.asarray(hidden, dtype=float))]
    if target_last is None:
        target_last = label == len(wo)-1
    inputs = [None] + ([target_value] if target_last else [])
    for fixed in inputs:
        updated = []
        for lo, hi, a, b in pieces:
            aa = wh @ a + (wi if fixed is None else 0.)
            bb = wh @ b + (0. if fixed is None else wi*fixed)
            cuts = [lo, hi]
            for slope, intercept in zip(aa, bb):
                if slope != 0:
                    root = -intercept/slope
                    stats['scalar_gate_roots'] += 1
                    if lo < root < hi:
                        cuts.append(float(root))
            cuts = sorted(set(cuts))
            for left, right in zip(cuts[:-1], cuts[1:]):
                active = aa*((left+right)/2)+bb > 0
                updated.append((left, right, aa*active, bb*active))
            stats['scalar_affine_steps'] += 1
        pieces = updated
    intervals = []
    for lo, hi, a, b in pieces:
        slopes = (wo[label]-wo) @ a
        offsets = (wo[label]-wo) @ b
        for other, (slope, intercept) in enumerate(zip(slopes, offsets)):
            if other == label:
                continue
            stats['scalar_readout_margins'] += 1
            if slope == 0:
                if intercept < 0 or (intercept == 0 and other < label):
                    hi = lo
                    break
            elif slope > 0:
                lo = max(lo, -intercept/slope)
            else:
                hi = min(hi, -intercept/slope)
            if hi <= lo:
                break
        if hi > lo:
            intervals.append((float(lo), float(hi)))
    stats['scalar_terminal_queries'] += 1
    stats['scalar_gate_pieces'] += len(pieces)
    stats['scalar_correct_intervals'] += len(intervals)
    return intervals


def terminal(model, hidden, u, q, label, limit, stats):
    """States indexed by remaining above-event count0/1."""
    intervals = decision_intervals(model, hidden, u, label, limit, stats)
    z = normal_mass(-limit, limit)
    lower = sum(normal_mass(lo, min(hi, u)) for lo, hi in intervals)
    upper = sum(normal_mass(max(lo, u), hi) for lo, hi in intervals)
    stats['normal_cdf_interval_queries'] += 2*len(intervals)
    answer = np.array([lower/(z*q), upper/(z*(1-q))])
    if np.any(answer < -1e-12) or np.any(answer > 1+1e-10):
        raise ArithmeticError('terminal probability outside unit interval')
    return answer


def possible_labels(wo):
    # First-index ties and the hidden origin make label0 always possible.
    result = []
    for j, row in enumerate(wo):
        if j == 0 or not any((k < j and np.all(other >= row)) or
                             np.all(other > row) for k, other in enumerate(wo) if k != j):
            result.append(j)
    return result


def grids(model, count, limit, stats, mesh_power=1.):
    wi, wh, wo = model
    bounds = [np.zeros(2)]
    for _ in range(len(wo)):
        bounds.append(np.maximum(wh, 0) @ bounds[-1]+limit*np.abs(wi))
    axes = [tuple(b*(1+1e-12)*np.linspace(0., 1., count)**mesh_power if b > 0 else np.array([0.])
                  for b in bound) for bound in bounds]
    states = [np.stack(np.meshgrid(*grid, indexing='ij'), axis=-1).reshape(-1, 2)
              for grid in axes]
    stats['hidden_grid_points_constructed'] += sum(map(len, states))
    return axes, states, bounds


def fixed_target(model, label, q, axes, states, ninput, limit, stats, deadline,
                 transition='quantile'):
    wi, wh, wo = model
    T = len(wo)
    alpha = ndtr(-limit)
    z = 1-2*alpha
    u = float(ndtri(alpha+z*q))
    if transition == 'quantile':
        x, w = leggauss(ninput)
        v, w = (x+1)/2, w/2
        below = ndtri(alpha+z*q*v)
        above = ndtri(alpha+z*(q+(1-q)*v))
        stats['conditional_inverse_normal_calls'] += 2*ninput+1
    else:
        from polynomial import conditional_expectations
        stats['conditional_inverse_normal_calls'] += 1
    stop = T-2 if label == T-1 else T-1
    values = np.array([terminal(model, h, u, q, label, limit, stats)
                       for h in states[stop]])
    values = values.reshape(len(axes[stop][0]), len(axes[stop][1]), 2)
    transition_error = 0.
    for t in reversed(range(stop)):
        if time.process_time() > deadline:
            raise TimeoutError('rank-conditioned cycle CPU limit')
        if t == label or transition == 'quantile':
            interp = RegularGridInterpolator(axes[t+1], values, bounds_error=True, method='linear')
            stats['interpolators_built'] += 1
        hbase = states[t] @ wh.T
        if t == label:
            coords = np.maximum(hbase+wi*u, 0)
            updated = interp(coords)
            stats['interpolation_queries'] += len(coords)
        else:
            m = T-t-int(label >= t)
            if transition == 'polynomial':
                low, high, error = conditional_expectations(model, states[t], axes[t+1], values,
                                                            u, q, limit, stats, deadline)
                expected = low, high
                transition_error = (1+error)*transition_error+error
            else:
                expected = []
                for samples in (below, above):
                    coords = np.maximum(hbase[:, None, :]+samples[None, :, None]*wi, 0)
                    coords = coords.reshape(-1, 2)
                    if len(coords) > 65536:
                        raise ArithmeticError('query batch exceeds fixed memory protocol')
                    evaluated = interp(coords).reshape(len(states[t]), ninput, 2)
                    expected.append(np.einsum('nqe,q->ne', evaluated, w))
                    stats['interpolation_queries'] += len(coords)
                    stats['scalar_transition_nodes'] += len(coords)
            updated = np.stack((expected[0][:, 0],
                                (m-1)/m*expected[0][:, 1]+expected[1][:, 0]/m), axis=1)
        if np.any(updated < -1e-12) or np.any(updated > 1+1e-10):
            raise ArithmeticError('transport probability outside unit interval')
        values = updated.reshape(len(axes[t][0]), len(axes[t][1]), 2)
    return float(values[0, 0, 1]), transition_error


def estimate(model, hidden_order=9, target_order=8, input_order=8, limit=6., seconds=110.,
             target_rule='jacobi', mesh_power=1., transition='quantile'):
    start = time.process_time()
    wi, wh, wo = (np.asarray(a, dtype=float) for a in model)
    T = len(wo)
    if wi.shape != (2,) or wh.shape != (2, 2) or wo.shape != (T, 2) or T < 3:
        raise ValueError('requires H2,T>=3 model')
    model = wi, wh, wo
    if transition not in ('quantile', 'polynomial'):
        raise ValueError('unknown transition method')
    stats = defaultdict(float)
    if mesh_power <= 0 or hidden_order < 2:
        raise ValueError('need positive mesh power and at least two grid nodes')
    axes, states, bounds = grids(model, hidden_order, limit, stats, mesh_power)
    if target_rule == 'jacobi':
        nodes, weights = roots_jacobi(target_order, 1, T-2)
        q = (nodes+1)/2
        weights = weights/weights.sum()
    elif target_rule == 'split_sign':
        nodes, base_weights = leggauss(target_order)
        q = np.concatenate(((nodes+1)/4, .5+(nodes+1)/4))
        weights = np.tile(base_weights/4, 2)*T*(T-1)*q**(T-2)*(1-q)
        if abs(weights.sum()-1) > 1e-12:
            raise ArithmeticError('target rule does not integrate Beta mass')
    else:
        raise ValueError('unknown target rule')
    stats['target_rule_nodes'] = len(q)
    labels = possible_labels(wo)
    contributions = {}
    transition_bound = 0.
    for label in labels:
        values = [fixed_target(model, label, qq, axes, states, input_order,
                               limit, stats, start+seconds, transition) for qq in q]
        contributions[label] = float(np.dot(weights, [v[0] for v in values])/T)
        transition_bound += float(np.dot(weights, [v[1] for v in values])/T)
    answer = sum(contributions.values())
    box_mass = float(np.exp(T*np.log1p(-2*ndtr(-limit))))
    return answer, dict(cpu_seconds=time.process_time()-start, work=dict(stats),
                        hidden_order=hidden_order, mesh_power=mesh_power,
                        target_order=target_order, target_rule=target_rule,
                        input_order=(input_order if transition == 'quantile' else None),
                        transition=transition,
                        transition_quadrature_only_bound=(transition_bound if transition == 'polynomial' else None),
                        limit=limit, possible_labels=labels, contributions=contributions,
                        stage_bounds=[b.tolist() for b in bounds],
                        omitted_input_mass_bound=1-box_mass,
                        input_tail_only_interval=[box_mass*answer, box_mass*answer+1-box_mass],
                        total_error_certified=False,
                        classification='deterministic quadrature and 2D interpolation with analytic scalar terminal')
