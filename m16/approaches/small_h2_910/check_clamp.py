"""Explicit range-clipping falsifier; no new model terminal geometry."""
from collections import defaultdict
import json
import pathlib
import time
import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln, stdtr
import backward
import doob

HERE = pathlib.Path(__file__).resolve().parent
BASE_INTERPOLATOR = backward.make_interpolator


def terminal_rank(b, k, stats):
    c = np.exp(gammaln((k+1)/2)-gammaln(k/2)) / np.sqrt(np.pi)
    def derivative(x):
        stats['rank_integrand_evaluations'] += 1
        return 2*c*(1+x*x)**(-(k+1)/2)*stdtr(k+1, x*np.sqrt((k+1)/(1+x*x)))
    values = []
    for x in b:
        integral, error = quad(derivative, 0., float(x), epsabs=1e-13, epsrel=1e-13)
        stats['rank_integral_calls'] += 1
        stats['max_rank_integral_reported_error'] = max(stats['max_rank_integral_reported_error'], error)
        F = .25+integral
        G = stdtr(k, x*np.sqrt(k))
        values.append((2*(G-F), F))
    values = np.array(values)
    stats['rank_q0_symmetry_error'] = float(np.max(abs(values[:, 0]-values[::-1, 0])))
    stats['rank_q1_symmetry_error'] = float(np.max(abs(values[::-1, 1]-(1-2*stdtr(k, b*np.sqrt(k))+values[:, 1]))))
    if np.any(values <= 0) or max(stats['rank_q0_symmetry_error'], stats['rank_q1_symmetry_error']) > 1e-12:
        raise ArithmeticError('rank identity failed')
    return values


def factory(clamped):
    def make(grid, values, method, stats):
        interpolator = BASE_INTERPOLATOR(grid, values, 'cubic', stats)
        def evaluate(coords):
            raw = interpolator(coords)
            stats['raw_minimum'] = min(stats.get('raw_minimum', float('inf')), float(np.min(raw)))
            stats['raw_maximum'] = max(stats.get('raw_maximum', -float('inf')), float(np.max(raw)))
            stats['raw_scalar_values'] += raw.size
            stats['raw_below_zero'] += int(np.count_nonzero(raw < 0))
            stats['raw_above_one'] += int(np.count_nonzero(raw > 1))
            stats['raw_below_minus_1e_10'] += int(np.count_nonzero(raw < -1e-10))
            stats['raw_above_one_plus_1e_10'] += int(np.count_nonzero(raw > 1+1e-10))
            if not clamped:
                return raw
            bounded = np.clip(raw, 0., 1.)
            correction = bounded-raw
            stats['changed_scalar_values'] += int(np.count_nonzero(correction))
            stats['absolute_correction_sum'] += float(np.sum(abs(correction)))
            stats['signed_correction_sum'] += float(np.sum(correction))
            stats['maximum_absolute_correction'] = max(stats['maximum_absolute_correction'], float(np.max(abs(correction))))
            return bounded
        return evaluate
    return make


def main():
    start = time.process_time()
    deadline = start+19
    model = backward.load(pathlib.Path('/workspace/clean-neurons/work/checkpoints/H2_T10_seed0.pth'))
    bound = backward.invariant_box(*model[:2])
    labels = backward.possible_labels(model[2])
    output = dict(complete=False, rows=[], model_forward_inputs=0, new_model_terminal_calls=0,
                  classification='explicit range-clamped numerical collocation, not an error certificate')
    try:
        for n in (5, 7, 9):
            grid = (np.linspace(0, bound[0], n), np.linspace(0, bound[1], n), np.linspace(-1, 1, n))
            with np.load(HERE/f'terminal_grid{n}.npz') as saved:
                cache = {j: saved[f'j{j}'] for j in labels}
            rank_stats = defaultdict(float)
            q = terminal_rank(grid[2], len(model[2])-2, rank_stats)
            rank = doob.RankRules(64, rank_stats)
            q_approx = rank.mass(grid[2], len(model[2])-2, 2)
            rules = doob.conditional_rules(grid[2], len(model[2]), 16, rank, rank_stats)
            for clamped in (False, True):
                clock = time.process_time()
                stats = defaultdict(float)
                contributions = {}
                doob.make_interpolator = factory(clamped)
                for j in labels:
                    if time.process_time() > deadline:
                        raise TimeoutError('19CPU-second deadline')
                    W = cache[j]/q[None, None, :, :]
                    contributions[j] = doob.transport(model, j, grid, W, rules, stats, method='cubic')[0]
                controls = defaultdict(float)
                constant = doob.transport(model, 0, grid, np.ones_like(cache[0]), rules, controls, method='cubic')[0]
                output['rows'].append(dict(grid=n, method='range_clamped_cubic' if clamped else 'original_cubic',
                    accuracy=sum(contributions.values()), contributions=contributions,
                    work=dict(stats), exact_rank_constant_control=constant, constant_control_error=constant-.1,
                    control_work=dict(controls), rank_work=dict(rank_stats),
                    terminal_q_max_difference_from_64_nodes=float(np.max(abs(q-q_approx))),
                    incremental_cpu_seconds=time.process_time()-clock))
        output['complete'] = True
    except (ValueError, ArithmeticError, TimeoutError) as error:
        output['failure'] = repr(error)
    finally:
        doob.make_interpolator = BASE_INTERPOLATOR
    output['cpu_seconds_excluding_imports'] = time.process_time()-start
    (HERE/'clamp_results.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output))

if __name__ == '__main__':
    main()
