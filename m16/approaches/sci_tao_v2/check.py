"""Bounded independent checks. Run from repo root; no test-model truth tuning."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import ndtr
import core as C


def check_model(model, name, n=64, grid=True):
    T = model[2].shape[0]
    out = {'model': name, 'H': len(model[0]), 'T': T, 'lines': 0}
    for key in ('mass_difference', 'grid_difference', 'new_fpe', 'old_charged_fpe', 'old_two_full_pass_proxy_fpe',
                'prefix_steps', 'suffix_steps', 'full_recompute_steps', 'cells', 'new_seconds', 'old_seconds', 'new_dense_dot_flops', 'old_dense_dot_flops'):
        out[key] = 0.
    for kind in (('12',) if T == 2 else ('12', '23')):
        y, a, b, tau = C.face_points(T, kind, C.sobol(T, n, 734))
        start = time.perf_counter()
        mass, cost, cells, work = C.walk_mass(model, y, a, b, tau, kind, details=True)
        out['new_seconds'] += time.perf_counter() - start
        start = time.perf_counter()
        old, old_cost, old_cells = C.geometry.walk_mass(model, y, a, b, tau, kind)
        out['old_seconds'] += time.perf_counter() - start
        delta = float(np.max(np.abs(mass - old)))
        assert delta < 2e-8, (name, kind, delta)
        assert np.array_equal(cells, old_cells), (name, kind, cells, old_cells)
        assert np.all(work['prefix_steps'] == np.minimum(a, b))
        assert np.all(work['suffix_steps'] <= T * cells - np.minimum(a, b))
        assert np.all(work['readout_vectors'] == 2 * cells)
        assert np.all(work['root_vectors'] == work['suffix_steps'])
        out['mass_difference'] = max(out['mass_difference'], delta)
        out['lines'] += n
        out['new_fpe'] += float(cost.sum())
        out['old_charged_fpe'] += float(old_cost.sum())
        out['old_two_full_pass_proxy_fpe'] += float(2 * old_cells.sum())
        for key in ('prefix_steps', 'suffix_steps'):
            out[key] += int(work[key].sum())
        out['full_recompute_steps'] += int(T * cells.sum())
        out['cells'] += int(cells.sum())
        H = len(model[0])
        out['new_dense_dot_flops'] += int((work['prefix_steps'].sum() + 2*work['suffix_steps'].sum()) * H*(2*H-1) + 2*cells.sum()*T*(2*H-1))
        out['old_dense_dot_flops'] += int(2*old_cells.sum()*(T*H*(2*H-1)+T*(2*H-1)))
        if grid:
            for j in range(min(4, n)):
                edges = np.linspace(-tau[j], tau[j], 65537)
                mids = (edges[:-1] + edges[1:]) / 2
                direction = np.zeros(T)
                direction[a[j]], direction[b[j]] = 1 / C.SQ2, -1 / C.SQ2
                logits = C.forward(model, y[j] + mids[:, None] * direction)
                target = np.where(mids < 0, b[j], a[j]) if kind == '23' else np.where(mids < 0, a[j], b[j])
                wrong = logits.argmax(axis=1) != target
                truth_grid = float(np.dot(wrong, np.diff(ndtr(edges))))
                # Any omitted crossing grid bin has at most max width in Gaussian mass.
                # There are at most T output-constraint crossings per gate cell.
                tolerance = (T * cells[j] + 1) * np.max(np.diff(ndtr(edges))) + 1e-9
                assert abs(truth_grid - mass[j]) <= tolerance, (name, kind, j, truth_grid, mass[j], tolerance)
                out['grid_difference'] = max(out['grid_difference'], abs(truth_grid - mass[j]))
    return out


def checkpoint_model(path):
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    return tuple(checkpoint[k].detach().numpy().astype(np.float64).reshape(-1) if k.endswith('ih_l0') else checkpoint[k].detach().numpy().astype(np.float64)
                 for k in ('rnn.weight_ih_l0', 'rnn.weight_hh_l0', 'linear.weight'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint')
    parser.add_argument('--output', default=str(Path(__file__).with_name('validation.json')))
    args = parser.parse_args()
    rng = np.random.default_rng(29341)
    models = []
    for H, T in ((1, 2), (2, 3), (4, 3), (5, 5), (8, 6), (12, 8), (16, 10), (32, 5)):
        model = (rng.normal(size=H), rng.normal(size=(H, H)) / np.sqrt(H), rng.normal(size=(T, H)))
        models.append((model, f'random_dev_H{H}_T{T}'))
    models.append(((np.zeros(3), np.zeros((3, 3)), np.zeros((4, 3))), 'all_zero_ties'))
    records = [check_model(m, name) for m, name in models]
    if args.checkpoint:
        base = checkpoint_model(args.checkpoint)
        for sigma in (0.0025, 0.02):
            # Identical first DEV copy as leaderboard.models(dev=True), without network loads.
            dev_rng = np.random.default_rng(1)
            model = tuple(w + sigma * np.sqrt((w*w).mean()) * dev_rng.standard_normal(w.shape) for w in base)
            records.append(check_model(model, f'dev_s{sigma}_0', n=256))
    # First-index ties must mean exact equality, not a fixed absolute tolerance.
    tiny_model = (np.ones(1), np.zeros((1, 1)), np.array([[0.], [1e-14], [0.]]))
    tiny_y, tiny_a, tiny_b, tiny_tau = np.array([[0., 0., 1.]]), np.array([0]), np.array([1]), np.array([.25])
    tiny_mass = C.walk_mass(tiny_model, tiny_y, tiny_a, tiny_b, tiny_tau, '23')[0][0]
    tiny_old = C.geometry.walk_mass(tiny_model, tiny_y, tiny_a, tiny_b, tiny_tau, '23')[0][0]
    assert abs(tiny_mass - (ndtr(.25) - .5)) < 1e-15
    assert abs(tiny_old - 2*tiny_mass) < 1e-15
    zero_mass, zero_cost, zero_cells = C.walk_mass(tiny_model, tiny_y, tiny_a, tiny_b, np.zeros(1), '23')
    assert zero_mass[0] == zero_cost[0] == zero_cells[0] == 0
    # Failure guard: no silent answer on incomplete integration.
    try:
        y, a, b, tau = C.face_points(3, '12', C.sobol(3, 16, 15))
        C.walk_mass(models[1][0], y, a, b, tau, '12', max_cells=0)
        raise AssertionError('Expected a cell-limit exception')
    except RuntimeError:
        pass
    Path(args.output).write_text(json.dumps({'records': records, 'cell_limit_guard': 'passed', 'zero_width_guard': 'passed', 'tiny_constant_logit_gap': {'new_mass': tiny_mass, 'old_mass': tiny_old, 'exact_mass': ndtr(.25)-.5}}, indent=2) + '\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
