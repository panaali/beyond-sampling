"""Gaussian integration of a bilinear continuation along scalar ReLU paths.

Exact polynomial moments on regular intervals; positive, bounded numerical
fallback on narrow intervals. All bounds exclude floating arithmetic errors.
"""
from functools import lru_cache
import math
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr


@lru_cache(None)
def fallback_rule():
    return leggauss(12)


def mix(corners, s, t):
    v00, v10, v01, v11 = corners
    return ((1-s)[..., None]*(1-t)[..., None]*v00[:, None, :]
            +s[..., None]*(1-t)[..., None]*v10[:, None, :]
            +(1-s)[..., None]*t[..., None]*v01[:, None, :]
            +s[..., None]*t[..., None]*v11[:, None, :])


def interval_integrals(lo, hi, corners, s, t, ds, dt, stats):
    """ds,dt are interpolation-coordinate changes from center to right edge."""
    center = (lo+hi)/2
    half = (hi-lo)/2
    magnitude = np.abs(center)*half+half*half/2
    endpoints = np.array([-1., 0., 1.])
    sampled = mix(corners, s[:, None]+ds[:, None]*endpoints,
                  t[:, None]+dt[:, None]*endpoints)
    a = sampled[:, 1]
    b = (sampled[:, 2]-sampled[:, 0])/2
    c = (sampled[:, 2]+sampled[:, 0])/2-sampled[:, 1]
    mass = np.empty(len(lo))
    positive = lo >= 0
    mass[positive] = ndtr(-lo[positive])-ndtr(-hi[positive])
    mass[~positive] = ndtr(hi[~positive])-ndtr(lo[~positive])
    phi_lo = np.exp(-lo*lo/2)/np.sqrt(2*np.pi)
    phi_hi = np.exp(-hi*hi/2)/np.sqrt(2*np.pi)
    raw_first = -phi_lo*np.expm1(-2*center*half)
    centered_first = raw_first-center*mass
    centered_second = (1+center*center)*mass-hi*phi_lo+lo*phi_hi
    values = a*mass[:, None]+b*(centered_first/half)[:, None]+c*(centered_second/half**2)[:, None]
    # These are stability screens, not certified floating-point error bounds.
    bad_moments = ((mass < 0) | (np.abs(centered_first) > half*mass*(1+1e-12))
                   | (centered_second < 0) | (centered_second > half**2*mass*(1+1e-12)))
    max_value = max(1., max(float(np.max(np.abs(v))) for v in corners))
    bad_value = (~np.isfinite(values).all(axis=1)
                 | (values.min(axis=1) < -1e-12*mass)
                 | (values.max(axis=1) > max_value*mass*(1+1e-12)))
    fallback = (magnitude < .125) | bad_moments | bad_value
    bounds = np.zeros(len(lo))
    nodes, weights = fallback_rule()
    indices = np.flatnonzero(fallback)
    for offset in range(0, len(indices), 4096):
        ids = indices[offset:offset+4096]
        ss = s[ids, None]+ds[ids, None]*nodes
        tt = t[ids, None]+dt[ids, None]*nodes
        point_values = mix(tuple(v[ids] for v in corners), ss, tt)
        xx = center[ids, None]+half[ids, None]*nodes
        density = np.exp(-xx*xx/2)/np.sqrt(2*np.pi)
        values[ids] = half[ids, None]*np.einsum('nqe,nq,q->ne', point_values, density, weights)
        M = magnitude[ids]
        bounds[ids] = (4*half[ids]*np.exp(-center[ids]**2/2)/np.sqrt(2*np.pi)
                       *max_value*np.exp(M)*M**11/math.factorial(11))
    stats['polynomial_intervals'] += len(lo)
    stats['polynomial_centered_moment_triples'] += len(lo)
    stats['polynomial_cdf_values'] += 2*len(lo)
    stats['polynomial_endpoint_density_exponentials'] += 2*len(lo)
    stats['polynomial_expm1_calls'] += len(lo)
    stats['polynomial_basis_mixes'] += 3*len(lo)+12*len(indices)
    stats['polynomial_positive_fallback_intervals'] += len(indices)
    stats['polynomial_positive_fallback_nodes'] += 12*len(indices)
    stats['polynomial_moment_screen_failures'] += int(np.count_nonzero(bad_moments))
    stats['polynomial_value_screen_failures'] += int(np.count_nonzero(bad_value))
    return values, bounds


def conditional_expectations(model, hidden, axes, values, u, q, limit, stats, deadline):
    """Integrate interpolated continuation under both conditional input laws."""
    wi, wh, _ = model
    alpha = ndtr(-limit)
    supports = (1-2*alpha)*np.array([q, 1-q])
    output = np.zeros((len(hidden), 2, 2))  # hidden, below/above, need flag
    errors = np.zeros((len(hidden), 2))
    for offset in range(0, len(hidden), 32):
        if time.process_time() > deadline:
            raise TimeoutError('polynomial transition CPU cap')
        h = hidden[offset:offset+32]
        base = h @ wh.T
        cuts = [np.full((len(h), 1), edge) for edge in (-limit, u, limit)]
        for axis in range(2):
            if wi[axis] != 0:
                crossing = (axes[axis][None, :]-base[:, axis, None])/wi[axis]
                cuts.append(np.clip(crossing, -limit, limit))
                stats['polynomial_grid_crossings_constructed'] += crossing.size
        ordered = np.sort(np.concatenate(cuts, axis=1), axis=1)
        stats['polynomial_sorts'] += len(h)
        rows, cols = np.nonzero(ordered[:, 1:] > ordered[:, :-1])
        lo, hi = ordered[rows, cols], ordered[rows, cols+1]
        center, half = (lo+hi)/2, (hi-lo)/2
        pre = base[rows]+center[:, None]*wi
        coords = np.maximum(pre, 0)
        indices, fractions, increments = [], [], []
        for axis in range(2):
            grid = axes[axis]
            if np.any(coords[:, axis] < grid[0]) or np.any(coords[:, axis] > grid[-1]):
                raise ArithmeticError('polynomial midpoint left proved hidden box')
            if len(grid) == 1:
                indices.append(np.zeros(len(lo), dtype=int))
                fractions.append(np.zeros(len(lo)))
                increments.append(np.zeros(len(lo)))
            else:
                cell = np.minimum(np.searchsorted(grid, coords[:, axis], side='right')-1, len(grid)-2)
                width = grid[cell+1]-grid[cell]
                indices.append(cell)
                fractions.append((coords[:, axis]-grid[cell])/width)
                increments.append(half*wi[axis]*(pre[:, axis] > 0)/width)
        ix, iy = indices
        jx = np.minimum(ix+1, len(axes[0])-1)
        jy = np.minimum(iy+1, len(axes[1])-1)
        corners = values[ix, iy], values[jx, iy], values[ix, jy], values[jx, jy]
        integrals, bound = interval_integrals(lo, hi, corners, *fractions, *increments, stats)
        side = (center > u).astype(int)
        np.add.at(output, (offset+rows, side), integrals)
        np.add.at(errors, (offset+rows, side), bound)
    output /= supports[None, :, None]
    errors /= supports[None, :]
    stats['polynomial_conditional_rows'] += 2*len(hidden)
    if np.any(output < -1e-11) or np.any(output > 1+1e-10):
        raise ArithmeticError('polynomial conditional probability outside unit interval')
    return output[:, 0], output[:, 1], float(np.max(errors))
