"""Exact nested subsampling of existing grid9 terminal values, no new kernels."""
from collections import defaultdict
import json
import pathlib
import time
import numpy as np
import backward
import doob
from check_clamp import terminal_rank, factory, BASE_INTERPOLATOR

HERE = pathlib.Path(__file__).resolve().parent


def main():
    started = time.process_time()
    deadline = started+19
    model = backward.load(pathlib.Path('/workspace/clean-neurons/work/checkpoints/H2_T10_seed0.pth'))
    bound = backward.invariant_box(*model[:2])
    labels = backward.possible_labels(model[2])
    full_grid = (np.linspace(0, bound[0], 9), np.linspace(0, bound[1], 9), np.linspace(-1, 1, 9))
    with np.load(HERE/'terminal_grid9.npz') as saved:
        full_cache = {j: saved[f'j{j}'] for j in labels}
    output = dict(complete=False, rows=[], model_forward_inputs=0, new_terminal_evaluations=0,
                  classification='deterministic anisotropic collocation diagnostic; no error certificate')
    try:
        for nh, nb in ((5, 9), (9, 5), (9, 9)):
            ih = slice(None, None, 2 if nh == 5 else 1)
            ib = slice(None, None, 2 if nb == 5 else 1)
            grid = (full_grid[0][ih], full_grid[1][ih], full_grid[2][ib])
            cache = {j: v[ih, ih, ib, :] for j, v in full_cache.items()}
            rank_stats = defaultdict(float)
            q = terminal_rank(grid[2], len(model[2])-2, rank_stats)
            rank = doob.RankRules(64, rank_stats)
            rules = doob.conditional_rules(grid[2], len(model[2]), 16, rank, rank_stats)
            for method in ('linear', 'range_clamped_cubic'):
                clock = time.process_time()
                stats = defaultdict(float)
                doob.make_interpolator = BASE_INTERPOLATOR if method == 'linear' else factory(True)
                contributions = {}
                for j in labels:
                    if time.process_time() > deadline:
                        raise TimeoutError('19CPU-second deadline')
                    contributions[j] = doob.transport(model, j, grid, cache[j]/q[None, None, :, :],
                        rules, stats, method='linear' if method == 'linear' else 'cubic')[0]
                controls = defaultdict(float)
                control = doob.transport(model, 0, grid, np.ones_like(cache[0]), rules, controls,
                    method='linear' if method == 'linear' else 'cubic')[0]
                output['rows'].append(dict(hidden_grid=nh, target_grid=nb, method=method,
                    accuracy=sum(contributions.values()), contributions=contributions,
                    constant_control=control, constant_error=control-.1,
                    work=dict(stats), control_work=dict(controls), rank_work=dict(rank_stats),
                    incremental_cpu_seconds=time.process_time()-clock))
        output['complete'] = True
    except (ValueError, ArithmeticError, TimeoutError) as error:
        output['failure'] = repr(error)
    finally:
        doob.make_interpolator = BASE_INTERPOLATOR
    output['cpu_seconds_excluding_imports'] = time.process_time()-started
    (HERE/'anisotropic_results.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output))

if __name__ == '__main__':
    main()
