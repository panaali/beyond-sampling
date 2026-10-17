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


def interval_geometry(model, hidden, axes, u, limit, stats, deadline):
    """Yield scalar pieces and four grid-corner indices in bounded batches."""
    wi, wh, _ = model
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
        side = (center > u).astype(int)
        corners = ((ix, iy), (jx, iy), (ix, jy), (jx, jy))
        yield offset+rows, side, lo, hi, corners, fractions, increments


def conditional_expectations(model, hidden, axes, values, u, q, limit, stats, deadline):
    """Integrate interpolated continuation under both conditional input laws."""
    supports = (1-2*ndtr(-limit))*np.array([q, 1-q])
    output = np.zeros((len(hidden), 2, 2))
    errors = np.zeros((len(hidden), 2))
    for rows, side, lo, hi, indices, fractions, increments in interval_geometry(
            model, hidden, axes, u, limit, stats, deadline):
        corners = tuple(values[ix, iy] for ix, iy in indices)
        integrals, bound = interval_integrals(lo, hi, corners, *fractions, *increments, stats)
        np.add.at(output, (rows, side), integrals)
        np.add.at(errors, (rows, side), bound)
    output /= supports[None, :, None]
    errors /= supports[None, :]
    stats['polynomial_conditional_rows'] += 2*len(hidden)
    if np.any(output < -1e-11) or np.any(output > 1+1e-10):
        raise ArithmeticError('polynomial conditional probability outside unit interval')
    return output[:, 0], output[:, 1], float(np.max(errors))


def conditional_matrices(model, hidden, axes, u, q, limit, stats, deadline):
    """Sparse interpolation kernels, independent of continuation or label."""
    supports = (1-2*ndtr(-limit))*np.array([q, 1-q])
    triplets = [([], [], []), ([], [], [])]
    errors = np.zeros((len(hidden), 2))
    for rows, side, lo, hi, indices, fractions, increments in interval_geometry(
            model, hidden, axes, u, limit, stats, deadline):
        corners = tuple(np.broadcast_to(np.eye(4)[k], (len(lo), 4)) for k in range(4))
        integrals, bound = interval_integrals(lo, hi, corners, *fractions, *increments, stats)
        columns = np.stack([ix*len(axes[1])+iy for ix, iy in indices], axis=1)
        for k in (0, 1):
            selected = side == k
            triplets[k][0].append(np.repeat(rows[selected], 4))
            triplets[k][1].append(columns[selected].ravel())
            triplets[k][2].append((integrals[selected]/supports[k]).ravel())
        # Four individually bounded basis integrals give a conservative sum
        # bound for the row operator acting on any continuation in[-1,1].
        np.add.at(errors, (rows, side), 4*bound)
    matrices = assemble_matrices(triplets, len(hidden), axes, stats)
    errors /= supports[None, :]
    return matrices, float(np.max(errors))


def assemble_matrices(triplets, nrows, axes, stats):
    from scipy.sparse import coo_matrix
    matrices = []
    for rows, columns, weights in triplets:
        data = np.concatenate(weights)
        matrix = coo_matrix((data, (np.concatenate(rows), np.concatenate(columns))),
                            shape=(nrows, len(axes[0])*len(axes[1]))).tocsr()
        matrix.eliminate_zeros()
        minimum = float(matrix.data.min()) if matrix.nnz else 0.
        defect = float(np.max(np.abs(np.asarray(matrix.sum(axis=1)).ravel()-1)))
        stats['polynomial_matrix_minimum_weight'] = min(stats['polynomial_matrix_minimum_weight'], minimum)
        stats['polynomial_matrix_maximum_row_defect'] = max(stats['polynomial_matrix_maximum_row_defect'], defect)
        stats['polynomial_sparse_triplets'] += len(data)
        stats['polynomial_sparse_nonzeros_constructed'] += matrix.nnz
        stats['polynomial_sparse_matrices_constructed'] += 1
        if minimum < -1e-12 or defect > 1e-10:
            raise ArithmeticError(f'sparse conditional kernel failed probability guards: minimum={minimum:.17g}, row_defect={defect:.17g}')
        matrices.append(matrix)
    return matrices


class StagePieces:
    """Full-input pieces reused across target quantiles inside one estimate."""

    def __init__(self, model, hidden, axes, limit, stats, deadline):
        self.axes, self.limit, self.nrows = axes, limit, len(hidden)
        fields = {name: [] for name in ('rows', 'lo', 'hi', 'columns', 's', 't', 'ds', 'dt', 'mass', 'bound')}
        for rows, _, lo, hi, indices, fractions, increments in interval_geometry(
                model, hidden, axes, -limit, limit, stats, deadline):
            corners = tuple(np.broadcast_to(np.eye(4)[k], (len(lo), 4)) for k in range(4))
            mass, bound = interval_integrals(lo, hi, corners, *fractions, *increments, stats)
            columns = np.stack([ix*len(axes[1])+iy for ix, iy in indices], axis=1)
            for key, value in zip(fields, (rows, lo, hi, columns, *fractions, *increments, mass, bound)):
                fields[key].append(value)
        for key, chunks in fields.items():
            setattr(self, key, np.concatenate(chunks))
        stats['polynomial_full_stage_caches_built'] += 1
        stats['polynomial_cached_full_intervals'] += len(self.lo)
        stats['polynomial_full_piece_array_bytes'] += sum(getattr(self, name).nbytes for name in fields)

    def matrices(self, u, q, stats, deadline):
        if time.process_time() > deadline:
            raise TimeoutError('piece-cache conditional kernel CPU cap')
        supports = (1-2*ndtr(-self.limit))*np.array([q, 1-q])
        crossing = (self.lo < u) & (self.hi > u)
        ids = np.flatnonzero(crossing)
        triplets = [([], [], []), ([], [], [])]
        errors = np.zeros((self.nrows, 2))
        old_mid = (self.lo[ids]+self.hi[ids])/2
        old_half = (self.hi[ids]-self.lo[ids])/2
        for side in (0, 1):
            complete = self.hi <= u if side == 0 else self.lo >= u
            triplets[side][0].append(np.repeat(self.rows[complete], 4))
            triplets[side][1].append(self.columns[complete].ravel())
            triplets[side][2].append((self.mass[complete]/supports[side]).ravel())
            np.add.at(errors[:, side], self.rows[complete], 4*self.bound[complete]/supports[side])
            if not len(ids):
                continue
            lo = self.lo[ids] if side == 0 else np.full(len(ids), u)
            hi = np.full(len(ids), u) if side == 0 else self.hi[ids]
            mid, half = (lo+hi)/2, (hi-lo)/2
            shift, shrink = (mid-old_mid)/old_half, half/old_half
            s, t = self.s[ids]+self.ds[ids]*shift, self.t[ids]+self.dt[ids]*shift
            ds, dt = self.ds[ids]*shrink, self.dt[ids]*shrink
            corners = tuple(np.broadcast_to(np.eye(4)[k], (len(ids), 4)) for k in range(4))
            mass, bound = interval_integrals(lo, hi, corners, s, t, ds, dt, stats)
            triplets[side][0].append(np.repeat(self.rows[ids], 4))
            triplets[side][1].append(self.columns[ids].ravel())
            triplets[side][2].append((mass/supports[side]).ravel())
            np.add.at(errors[:, side], self.rows[ids], 4*bound/supports[side])
            stats['polynomial_partial_target_intervals'] += len(ids)
        stats['polynomial_stage_piece_cache_queries'] += 1
        matrices = assemble_matrices(triplets, self.nrows, self.axes, stats)
        return matrices, float(np.max(errors))
