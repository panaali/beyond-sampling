"""Bounded T8 diagnostics; no shared-code modifications or online dependencies with --weights."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import torch


def orth(a):
    u, s, _ = np.linalg.svd(a, full_matrices=False)
    return u[:, s > max(s[0], 1) * 1e-10]


def gates(h):
    return (h > 0).astype(np.int64) @ (1 << np.arange(h.shape[-1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=Path)
    p.add_argument('--output', type=Path, default=Path(__file__).with_name('results.json'))
    p.add_argument('--pilot', type=int, default=10000)
    p.add_argument('--n', type=int, default=40000)
    p.add_argument('--seed', type=int, default=82026)
    p.add_argument('--mode-selection', choices=['dominant', 'slowest'], default='dominant')
    a = p.parse_args()
    start, cpu = time.perf_counter(), time.process_time()
    if a.weights:
        state = torch.load(a.weights, weights_only=True, map_location='cpu')
        weight_hash = hashlib.sha256(a.weights.read_bytes()).hexdigest()
    else:
        from alg_zoo import example_2nd_argmax
        state = example_2nd_argmax().state_dict()
        weight_hash = 'online example_2nd_argmax; array hash below'
    wi = state['rnn.weight_ih_l0'].numpy()[:, 0].astype(float)
    wh = state['rnn.weight_hh_l0'].numpy().astype(float)
    wo = state['linear.weight'].numpy().astype(float)
    assert wh.shape == (16, 16) and wo.shape == (10, 16)
    counts = {'model_sequences': 0, 'hidden_steps': 0, 'output_rows': 0,
              'projection_multiply_add_flops': 0, 'gaussian_variates': 0,
              'eigendecompositions': 0, 'svd_decompositions': 0, 'phase_shuffle_rows': 0}
    rng = np.random.default_rng(a.seed)

    def generate(n):
        x = rng.standard_normal((n, 10))
        counts['gaussian_variates'] += n * 10
        hs = np.zeros((n, 11, 16))
        for t in range(10):
            hs[:, t+1] = np.maximum(hs[:, t] @ wh.T + x[:, t, None] * wi, 0)
        counts['model_sequences'] += n
        counts['hidden_steps'] += n * 10
        return x, hs

    xp, hp = generate(a.pilot)
    unique, freq = np.unique(gates(hp[:, 6:11].reshape(-1, 16)), return_counts=True)
    selected = np.argsort(-freq)[:5]
    x, hs = generate(a.n)
    final = hs[:, -1]
    codes, pcodes = gates(final), gates(hp[:, -1])
    y = np.argsort(x, axis=1)[:, -2]
    pred = (final @ wo.T).argmax(1)
    counts['output_rows'] += a.n
    records = []
    for idx in selected:
        code = int(unique[idx])
        active = np.flatnonzero(code & (1 << np.arange(16)))
        ev, v = np.linalg.eig(wh[np.ix_(active, active)])
        counts['eigendecompositions'] += 1
        inv = np.linalg.inv(v)
        complex_pairs = sorted(np.flatnonzero(ev.imag > 1e-7), key=lambda j: np.angle(ev[j]))
        mask = (codes == code) & (x.max(1) > 0)
        h, yy, pp = final[mask], y[mask], pred[mask]
        age = 9 - x[mask].argmax(1)
        row = {'gate_code': code, 'active': active.tolist(),
               'pilot_share_steps_6_10': float(freq[idx] / (a.pilot * 5)),
               'n_final_positive_max': len(h), 'final_share': float((codes == code).mean()),
               'eigenbasis_condition': float(np.linalg.cond(v)),
               'complex_periods': [float(2*np.pi/np.angle(ev[j])) for j in complex_pairs]}
        if not complex_pairs or len(h) < 100:
            row['skip'] = 'no complex pair or insufficient final states'
            records.append(row)
            continue
        j = max(complex_pairs, key=lambda k: abs(ev[k])) if a.mode_selection == 'dominant' else complex_pairs[0]
        z = h[:, active] @ inv[j]
        unit = z / np.maximum(np.abs(z), 1e-15)
        theta = np.angle(ev[j])
        bins = []
        for k in range(10):
            part = unit[age == k]
            if len(part):
                mean = part.mean()
                bins.append({'age': k, 'n': len(part), 'angle_deg': float(np.angle(mean)*180/np.pi),
                             'concentration': float(abs(mean))})
        supported = [b for b in bins if b['n'] >= 50]
        slope = None
        if len(supported) >= 3:
            slopes = np.polyfit([b['age'] for b in supported],
                                np.unwrap([b['angle_deg']*np.pi/180 for b in supported]), 1)
            slope = float(slopes[0]*180/np.pi)
        aligned = float(abs(np.mean(unit * np.exp(-1j*theta*age))))
        shuffles = [float(abs(np.mean(unit*np.exp(-1j*theta*rng.permutation(age))))) for _ in range(199)]
        counts['phase_shuffle_rows'] += len(age)*199
        row['phase'] = {'predicted_deg_per_age': float(theta*180/np.pi), 'observed_deg_per_age': slope,
                        'age_bins': bins, 'alignment': aligned, 'shuffle_median': float(np.median(shuffles)),
                        'shuffle_p': float((1+sum(s >= aligned for s in shuffles))/200)}
        if len(complex_pairs) < 3:
            row['fast_skip'] = 'fewer than three complex pairs'
            records.append(row)
            continue
        fast_indices = (sorted([q for q in complex_pairs if np.angle(ev[q]) > theta + 1e-8], key=lambda q: -abs(ev[q]))[:2]
                        if a.mode_selection == 'dominant' else complex_pairs[1:3])
        if len(fast_indices) < 2:
            row['fast_skip'] = 'fewer than two faster pairs'
            records.append(row)
            continue
        row['selected_fast_periods'] = [float(2*np.pi/np.angle(ev[q])) for q in fast_indices]
        raw = np.column_stack([part for q in fast_indices for part in (v[:, q].real, v[:, q].imag)])
        qfast = orth(raw)
        qslow = orth(np.column_stack((v[:, j].real, v[:, j].imag)))
        counts['svd_decompositions'] += 2
        leverage = np.sum(qfast*qfast, axis=1)
        coord_pairs = list(itertools.combinations(range(len(active)), 2))
        target_pair = tuple(int(np.where(active == n)[0][0]) for n in (3,13)) if {3,13}.issubset(active) else None
        target_leverage = float(leverage[list(target_pair)].mean()) if target_pair else None
        row['orthogonal'] = {'rank': qfast.shape[1], 'leverage_by_active_neuron': leverage.tolist(),
                             'n3_n13_mean_leverage': target_leverage,
                             'pair_count': len(coord_pairs),
                             'n3_n13_leverage_rank': 1+sum(float(leverage[list(pair)].mean()) > target_leverage+1e-12 for pair in coord_pairs) if target_pair else None,
                             'slow_fast_overlap_squared': float(np.sum((qslow.T@qfast)**2)),
                             'note': 'Overlap 0 means orthogonal spans; maximum is 2.'}
        mean = hp[pcodes == code, -1].mean(0)
        centered = h - mean
        baseline = pp == yy

        def ablate(q):
            removed = centered @ q @ q.T
            hh = h - removed
            new = (hh @ wo.T).argmax(1) == yy
            delta = baseline.astype(float) - new.astype(float)
            counts['output_rows'] += len(h)
            counts['projection_multiply_add_flops'] += 4*len(h)*16*q.shape[1]
            return {'accuracy': float(new.mean()), 'accuracy_drop_pp': float(100*delta.mean()),
                    'paired_se_pp': float(100*delta.std(ddof=1)/np.sqrt(len(delta))),
                    'removed_centered_energy_fraction': float(np.sum(removed**2)/np.sum(centered**2)),
                    'negative_state_fraction': float((hh < -1e-10).mean())}

        qf = np.zeros((16, qfast.shape[1])); qf[active] = qfast
        qs = np.zeros((16, qslow.shape[1])); qs[active] = qslow
        fast_result = ablate(qf)
        slow_result = ablate(qs)
        controls = []
        for _ in range(32):
            random = rng.standard_normal((len(active), qfast.shape[1]))
            counts['gaussian_variates'] += random.size
            qlocal = orth(random)
            counts['svd_decompositions'] += 1
            q = np.zeros_like(qf); q[active] = qlocal
            controls.append(ablate(q))
        pair_results = []
        for pair in coord_pairs:
            q = np.eye(16)[:, active[list(pair)]]
            pair_results.append({'neurons': active[list(pair)].tolist(), **ablate(q)})
        row['ablation'] = {'baseline_accuracy': float(baseline.mean()), 'fast': fast_result,
                           'slow': slow_result, 'random_rank_matched': controls, 'neuron_pairs': pair_results}
        records.append(row)
    counts['rnn_arithmetic_flops'] = counts['hidden_steps'] * (2*16*16 + 3*16)
    counts['readout_arithmetic_flops'] = counts['output_rows'] * 2*16*10
    counts['gaussian_generation_flop_convention'] = 20 * counts['gaussian_variates']
    counts['counted_arithmetic_flops'] = sum(counts[k] for k in ('rnn_arithmetic_flops', 'readout_arithmetic_flops', 'projection_multiply_add_flops', 'gaussian_generation_flop_convention'))
    result = {'mode_selection': a.mode_selection, 'seed': a.seed, 'pilot_n': a.pilot, 'evaluation_n': a.n,
              'weights_file_sha256': weight_hash,
              'weights_array_sha256': hashlib.sha256(wi.tobytes()+wh.tobytes()+wo.tobytes()).hexdigest(),
              'overall_accuracy': float((pred == y).mean()), 'gate_checks': records, 'compute': counts,
              'accounting_limits': 'Arithmetic subtotal excludes sort/argmax/comparison/reduction, complex phase/shuffle, eig/SVD/inverse FLOPs. Their operation counts and total CPU time are reported; subtotal is not a leaderboard cost.',
              'cpu_seconds': time.process_time()-cpu, 'wall_seconds': time.perf_counter()-start}
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('overall_accuracy','cpu_seconds','wall_seconds','compute')}, indent=2))

if __name__ == '__main__':
    main()
