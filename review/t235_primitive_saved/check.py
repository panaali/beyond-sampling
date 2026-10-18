"""Saved-only independent T235 review; never import the author primitive."""
import hashlib
import json
import math
import resource
import time
import traceback
from fractions import Fraction as Q
from pathlib import Path

HERE = Path(__file__).resolve().parent
START = time.process_time()
CUTS = (Q(0), Q(1, 4), Q(1, 2))


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_pins(mapping):
    assert mapping
    for path, expected in mapping.items():
        assert digest(path) == expected, path


def cdf(x, c, u):
    # Integrate interval densities through u, rather than select a knot by bisect.
    return sum((c[i + 1] - c[i]) * max(Q(0), min(Q(1), (u - x[i]) / (x[i + 1] - x[i])))
               for i in range(len(x) - 1))


def inverse(x, c, q):
    # A repeated ordinate uses the explicitly frozen rightmost endpoint policy.
    knots = [i for i, height in enumerate(c) if height == q]
    if knots:
        return x[max(knots)]
    candidates = [i for i in range(len(c) - 1) if c[i] < q < c[i + 1]]
    assert len(candidates) == 1
    i = candidates[0]
    return ((c[i + 1] - q) * x[i] + (q - c[i]) * x[i + 1]) / (c[i + 1] - c[i])


def guard_reason(spec):
    op = spec['op']
    if op == 'construct':
        x, c = tuple(map(Q, spec['x'])), tuple(map(Q, spec['c']))
        invalid = (len(x) != len(c) or len(x) < 2 or
                   any(a >= b for a, b in zip(x, x[1:])) or
                   c[0] != 0 or c[-1] != 1 or any(a > b for a, b in zip(c, c[1:])))
        assert invalid
        return 'ValueError'
    if op in ('float_construct', 'float_cdf'):
        return 'TypeError'
    if op == 'inverse':
        assert not 0 <= Q(spec['q']) <= 1
    elif op == 'conditional':
        assert not 0 <= spec['bin'] < 4 or not 0 < Q(spec['v']) < 1
    elif op == 'zero_stratum':
        assert cdf((Q(1), Q(2)), (Q(0), Q(1)), Q(0)) == 0
    else:
        raise AssertionError(op)
    return 'ValueError'


def run(report):
    scope = read(HERE / 'scope.json')
    verify_pins(scope['pins'])
    gate = read(HERE / 'gate.json')
    assert gate['status'] == 'ready' and gate['review_publication']
    review_path = gate['source_review']
    review = read(review_path)
    assert review['status'] == 'passed' and review['execution_authorized'] is True
    required = {str(HERE / 'check.py'), str(HERE / 'scope.json')}
    assert required <= set(review['pins']) and required | {review_path} <= set(gate['pins'])
    verify_pins(gate['pins']); verify_pins(review['pins'])
    author = Path(scope['author'])
    p = read(author / 'protocol.json'); g = read(author / 'gate.json')
    s = read(g['source_review'])
    assert g['status'] == 'ready' and g['review_publication']
    assert s['status'] == 'passed' and s['execution_authorized'] is True
    verify_pins(p['pins']); verify_pins(g['pins']); verify_pins(s['pins'])
    f = read(author / 'fixtures.json'); raw = read(author / 'output/results.json')
    assert raw['status'] == 'complete' and raw['pins'] == g['pins']
    assert raw['model_grid_face_rng_calls'] == 0
    assert math.isfinite(raw['cpu']) and raw['cpu'] >= 0
    assert len(f['grids']) == len(raw['rows']) == p['grids'] == 8
    assert len(f['guards']) == len(raw['guard_rows']) == p['guards'] == 16
    assert f['q'] == ['0', '1/4', '1/2', '3/4', '1']
    assert f['conditional_v'] == ['1/4', '1/2', '3/4']
    assert len({v['name'] for v in f['grids'] + f['guards']}) == 24
    expected_events = [('start', None)]
    ncdf = ninverse = nconditional = 0
    report['rows'] = []
    for spec, row in zip(f['grids'], raw['rows'], strict=True):
        assert time.process_time() - START < 19
        assert row['fixture'] == spec and row['name'] == spec['name'] and row['status'] == 'passed'
        x, c = tuple(map(Q, spec['x'])), tuple(map(Q, spec['c']))
        assert len(x) == len(c) >= 2 and c[0] == 0 and c[-1] == 1
        assert all(a < b for a, b in zip(x, x[1:])) and all(a <= b for a, b in zip(c, c[1:]))
        ends = [Q(0)] + [cdf(x, c, u) for u in CUTS] + [Q(1)]
        masses = [ends[j + 1] - ends[j] for j in range(4)]
        assert list(map(Q, row['bins'])) == list(map(Q, spec['bins'])) == masses
        assert min(masses) >= 0 and sum(masses) == 1
        probes = list(x + CUTS)
        assert [Q(v['u']) for v in row['cdf']] == probes
        for u, value in zip(probes, row['cdf'], strict=True):
            assert Q(value['value']) == cdf(x, c, u)
        ncdf += len(probes)
        assert [Q(v['q']) for v in row['inverse']] == list(map(Q, f['q']))
        for qs, expected, value in zip(f['q'], spec['inverse'], row['inverse'], strict=True):
            assert Q(value['u']) == Q(expected) == inverse(x, c, Q(qs))
        ninverse += len(row['inverse'])
        intended = [(j, Q(v)) for j, mass in enumerate(masses) if mass > 0 for v in f['conditional_v']]
        assert [(v['bin'], Q(v['v'])) for v in row['conditional']] == intended
        for (j, v), value in zip(intended, row['conditional'], strict=True):
            q = ends[j] + masses[j] * v
            u = inverse(x, c, q)
            assert Q(value['probability']) == masses[j] and Q(value['q']) == q and Q(value['u']) == u
            assert sum(u > cut for cut in CUTS) == j
        nconditional += len(intended)
        report['rows'].append({'name':spec['name'], 'bins':list(map(str,masses)), 'cdf':len(probes), 'inverse':5, 'conditional':len(intended)})
        expected_events.extend([('grid_start', spec['name']), ('grid_complete', spec['name'])])
    for spec, row in zip(f['guards'], raw['guard_rows'], strict=True):
        assert row['fixture'] == spec and row['name'] == spec['name'] and row['status'] == 'passed'
        expected = guard_reason(spec)
        assert row['caught'] == spec['error'] == expected
        assert row['exception'].startswith(expected + '(')
        expected_events.extend([('guard_start', spec['name']), ('guard_complete', spec['name'])])
    expected_events.append(('complete', None))
    events = [json.loads(line) for line in (author / 'output/progress.jsonl').read_text().splitlines()]
    assert [(v['event'], v.get('name')) for v in events] == expected_events
    assert len(events) == p['progress_events'] == 50
    cpus = [v['cpu'] for v in events]
    assert all(math.isfinite(v) and v >= 0 for v in cpus)
    assert all(a <= b for a,b in zip(cpus,cpus[1:]))
    assert cpus[-2] <= raw['cpu'] <= cpus[-1]
    assert ninverse == 40 and nconditional == raw['conditional_calls'] == p['conditional_calls'] == 78
    report.update(status='passed', grids=8, guards=16, cdf_probes=ncdf, inverse_probes=ninverse,
                  conditional_records=nconditional, events=len(events), author_cpu=raw['cpu'],
                  input_pins=scope['pins'], author_calls=0,
                  limitation='Saved exact synthetic arithmetic only; no actual face grid, finite NumPy parity, Gaussian probability, cost or estimator claim.')


if __name__ == '__main__':
    target = HERE / 'results.json'
    assert not target.exists(), 'Preserve prior attempts'
    result = {'status':'started'}
    target.write_text(json.dumps(result) + '\n')
    try:
        run(result)
    except BaseException as exc:
        result.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        result['review_cpu'] = time.process_time() - START
        result['maxrss_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        temporary = HERE / 'results.tmp'
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        temporary.replace(target)
