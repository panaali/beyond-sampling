"""Charged terminal-curvature mesh monitor; no reference accuracy is used."""
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr, ndtri
from engine import batched_terminal, possible_labels


def terminal_mesh(mode):
    if mode not in ('curvature', 'reversed', 'gate'):
        raise ValueError(mode)

    def build(model, axes, states, limit, stats):
        start = time.process_time()
        stage = len(model[2])-1
        labels = [j for j in possible_labels(model[2]) if j != stage]
        if mode == 'gate':
            newaxes = []
            scales = []
            wi, wh, _ = model
            for d in range(2):
                bound = axes[stage][d][-1]
                valid = (wi != 0) & (wh[:, d] != 0)
                if not np.any(valid) or bound == 0:
                    newaxes.append(axes[stage][d])
                    scales.append(None)
                    continue
                scale = float(np.min(abs(wi[valid]/wh[valid, d])))
                axis = scale*np.expm1(np.linspace(0, 1, len(axes[stage][d]))*np.log1p(bound/scale))
                axis[-1] = bound
                if np.any(np.diff(axis) <= 0):
                    raise ArithmeticError('gate mesh not strictly ordered')
                newaxes.append(axis)
                scales.append(scale)
                stats['gate_mesh_scalar_transcendentals'] += 1+len(axis)
                stats['gate_mesh_scale_ratios'] += int(np.count_nonzero(valid))
            axes, states = list(axes), list(states)
            axes[stage] = tuple(newaxes)
            states[stage] = np.stack(np.meshgrid(*newaxes, indexing='ij'), axis=-1).reshape(-1, 2)
            stats['hidden_grid_points_constructed'] += len(states[stage])
            return axes, states, dict(mode=mode, changed_stage=stage, scales=scales,
                monitor=[dict(axis=a.tolist()) for a in newaxes], node_count=len(states[stage]))
        nodes, weights = leggauss(4)
        q = np.r_[(nodes+1)/4, .5+(nodes+1)/4]
        weights = np.tile(weights/4, 2)*len(model[2])*(len(model[2])-1)*q**(len(model[2])-2)*(1-q)
        alpha = ndtr(-limit)
        monitor = []
        newaxes = []
        for d in range(2):
            bound = axes[stage][d][-1]
            if bound == 0:
                newaxes.append(axes[stage][d])
                monitor.append(dict(degenerate=True))
                continue
            edges = bound*np.linspace(0, 1, 33)**2
            mid = (edges[:-1]+edges[1:])/2
            other = axes[stage][1-d][-1]*np.linspace(0, 1, 9)**2
            # Every bin's left, midpoint and right, each with nine other-axis values.
            positions = np.stack((edges[:-1], mid, edges[1:]), axis=1)
            grid = np.empty((32, 3, 9, 2))
            grid[..., d] = positions[..., None]
            grid[..., 1-d] = other
            grid = grid.reshape(-1, 2)
            score = np.zeros(32)
            cache = {}
            for label in labels:
                for qq, weight in zip(q, weights):
                    u = ndtri(alpha+(1-2*alpha)*qq)
                    vals = batched_terminal(model, grid, u, qq, label, limit, stats, cache).reshape(32, 3, 9, 2)
                    residual = abs(vals[:, 1]-(vals[:, 0]+vals[:, 2])/2)
                    score += weight*residual.mean(axis=(1, 2))/len(labels)
            density = np.cbrt(score)
            if density.sum() == 0:
                density[:] = 1
            density /= density.sum()
            if mode == 'reversed':
                density = density[::-1]
            mass = .5/32+.5*density
            cumulative = np.r_[0, np.cumsum(mass)]
            cumulative[-1] = 1.
            axis = np.interp(np.linspace(0, 1, len(axes[stage][d])), cumulative, edges)
            if np.any(np.diff(axis) <= 0):
                raise ArithmeticError('mesh not strictly ordered')
            newaxes.append(axis)
            stats['mesh_monitor_residuals'] += len(labels)*len(q)*32*9*2
            stats['mesh_monitor_quantile_interpolations'] += len(axis)
            monitor.append(dict(bin_edges=edges.tolist(), midpoint_residual=score.tolist(),
                                bin_mass=mass.tolist(), axis=axis.tolist(),
                                mean_midpoint_residual=float(score.mean())))
        axes = list(axes)
        states = list(states)
        axes[stage] = tuple(newaxes)
        states[stage] = np.stack(np.meshgrid(*newaxes, indexing='ij'), axis=-1).reshape(-1, 2)
        stats['hidden_grid_points_constructed'] += len(states[stage])
        stats['mesh_monitor_cpu_seconds'] += time.process_time()-start
        return axes, states, dict(mode=mode, changed_stage=stage, monitor=monitor,
                                  node_count=len(states[stage]))
    return build
