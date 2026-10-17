"""Fixed-query and full existing-grid equivalence/cost checks for T59."""
from collections import defaultdict
import hashlib
import itertools
import json
import pathlib
import resource
import time
import numpy as np
from backward import FUND, load, invariant_box, possible_labels, target_probability
from batch_terminal import BatchTerminal
from positive import rule

HERE = pathlib.Path(__file__).resolve().parent


def add_work(stats, work):
    for key, value in work.items():
        if isinstance(value, (int, float)) and key not in ('partition_mass', 'prefix_dimension', 'target_index'):
            stats[key] += value


def main():
    started = time.process_time()
    deadline = started + 115
    model = load(pathlib.Path('/workspace/clean-neurons/work/checkpoints/H2_T10_seed0.pth'))
    bound = invariant_box(*model[:2])
    output = dict(complete=False, model_forward_inputs=0, validation=[], full_grid={},
                  source_sha256={name: hashlib.sha256((FUND/name).read_bytes()).hexdigest()
                                 for name in ('target_tail.py', 'tail.py', 'moments.py', 'positive.py')})
    def guard():
        if time.process_time() > deadline:
            raise TimeoutError('115CPU-second deadline')
    def save():
        output['cpu_seconds_excluding_imports'] = time.process_time() - started
        output['max_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (HERE/'batch_results.json').write_text(json.dumps(output, indent=2)+'\n')
    try:
        states = [np.zeros(2), np.array([bound[0], 0]), np.array([0, bound[1]]), bound*np.array([.37, .61])]
        cases = list(itertools.product(states, (-1., 0., 1.), (0, 1), (0, 5, 8, 9)))
        tied = (model[0], model[1], np.zeros_like(model[2]))
        tiecases = list(itertools.product(states[:2], (-1., 0., 1.), (0, 1), (0, 9)))
        for name, weights, queries in [('trained96', model, cases), ('first_index_ties24', tied, tiecases)]:
            scalar_values = []
            scalar_work = defaultdict(float)
            rule.cache_clear()
            clock = time.process_time()
            for a, b, count, j in queries:
                guard()
                p, stats = target_probability(weights, a, b, count, j)
                scalar_values.append(p)
                add_work(scalar_work, stats)
            scalar_cpu = time.process_time()-clock
            rule.cache_clear()
            batch = BatchTerminal(weights)
            batch_values = []
            for a, b, count, j in queries:
                guard()
                batch_values.append(batch.probability(a, b, count, j))
            differences = abs(np.array(batch_values)-scalar_values)
            assert np.max(differences) <= 1e-12
            assert batch.probability(states[0], 0., 2, 9) == 0.
            if name.startswith('first'):
                assert all(p == 0. for p, query in zip(batch_values, queries) if query[-1] == 9)
            output['validation'].append(dict(name=name, queries=len(queries),
                max_absolute_difference=float(np.max(differences)), scalar_cpu_seconds=scalar_cpu,
                scalar_work=dict(scalar_work), batch=batch.report()))
            save()
        # New cold batch: no geometry or moment cache from the validation phase.
        rule.cache_clear()
        batch = BatchTerminal(model)
        grid = (np.linspace(0, bound[0], 9), np.linspace(0, bound[1], 9), np.linspace(-1, 1, 9))
        maximum = 0.
        count_queries = 0
        clock = time.process_time()
        with np.load(HERE/'terminal_grid9.npz') as reference:
            for j in possible_labels(model[2]):
                reference_values = reference[f'j{j}']
                for ix, x in enumerate(grid[0]):
                    for iy, y in enumerate(grid[1]):
                        for ib, b in enumerate(grid[2]):
                            for count in (0, 1):
                                guard()
                                p = batch.probability((x, y), float(b), count, j)
                                difference = abs(p-float(reference_values[ix, iy, ib, count]))
                                maximum = max(maximum, difference)
                                count_queries += 1
                                if difference > 1e-12:
                                    raise ArithmeticError(f'grid mismatch {(ix,iy,ib,count,j)}: {difference}')
                output['full_grid'] = dict(batch=batch.report(), queries=count_queries,
                    maximum_absolute_difference=maximum, total_driver_cpu_seconds=time.process_time()-clock)
                save()
        output['complete'] = True
    except (ArithmeticError, ValueError, TimeoutError, AssertionError) as error:
        output['failure'] = repr(error)
    save()
    print(json.dumps(output))

if __name__ == '__main__':
    main()
