"""Single frozen synthetic attempt. No candidate import before metadata gates."""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from fractions import Fraction as Q

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'output'
OWNED = False
D = {}
START = 0.0


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def encode(x):
    if isinstance(x, Q):
        return {'q': [str(x.numerator), str(x.denominator)]}
    if isinstance(x, dict):
        return {str(k): encode(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [encode(v) for v in x]
    return x


def save():
    if not OWNED:
        return
    D['cpu_seconds'] = time.process_time()-START
    D['max_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    p = OUT/'results.tmp'
    with p.open('x') as f:
        json.dump(encode(D), f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
    os.replace(p, OUT/'results.json')


def event(kind, **kw):
    with (OUT/'progress.jsonl').open('a') as f:
        f.write(json.dumps(dict(kind=kind, cpu_seconds=time.process_time()-START, **kw), sort_keys=True)+'\n')


def verify(pins):
    assert pins
    for p, s in pins.items():
        assert isinstance(s, str) and len(s) == 64 and sha(p) == s, p


def main():
    global OWNED, START, D
    START = time.process_time()
    p = json.loads((ROOT/'protocol.json').read_text())
    g = json.loads((ROOT/'gate.json').read_text())
    assert not OUT.exists(), 'immutable output exists'
    verify(p['pins']); verify(g['pins'])
    prior = json.loads(Path(p['prior_written_review']).read_text())
    assert prior['complete'] is True and prior['pass_all'] is True
    verify(prior['pins'])
    for name in ('algebra.py', 'fixtures.py', 'run.py', 'launch.sh', 'PROTOCOL.md'):
        assert str(ROOT/name) in p['pins'] and str(ROOT/name) in g['pins']
    assert str(ROOT/'protocol.json') in g['pins']
    assert g['source_review'] in g['pins']
    r = json.loads(Path(g['source_review']).read_text())
    assert all(r[k] is True for k in ('complete', 'pass_all', 'execution_authorized'))
    verify(r['pins'])
    for name in ('algebra.py', 'fixtures.py', 'run.py', 'launch.sh', 'PROTOCOL.md', 'protocol.json'):
        assert str(ROOT/name) in r['pins']
    for key in ('source_review_publication', 'execution_scope_publication'):
        assert isinstance(g[key], str) and len(g[key]) >= 8 and all(c in '0123456789abcdef' for c in g[key])
    assert sys.executable == p['invoked_python']
    assert str(Path(sys.executable).resolve()) == p['resolved_python']
    assert sha(p['resolved_python']) == p['python_sha256']
    assert resource.getrlimit(resource.RLIMIT_CPU) == (16, 20)
    assert resource.getrlimit(resource.RLIMIT_AS) == (2147483648, 2147483648)
    assert all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'))
    OUT.mkdir(); OWNED = True
    D = dict(complete=False, pass_all=False, protocol_sha256=sha(ROOT/'protocol.json'),
             gate_sha256=sha(ROOT/'gate.json'), rows=[], solve_attempts=0, solve_returns=0,
             runtime=dict(invoked=sys.executable, resolved=p['resolved_python'], version=sys.version,
                          cpu_limits=list(resource.getrlimit(resource.RLIMIT_CPU)),
                          address_space_limits=list(resource.getrlimit(resource.RLIMIT_AS))))
    save(); event('started', protocol_sha256=D['protocol_sha256'], gate_sha256=D['gate_sha256'])
    import algebra as a
    from fixtures import cases
    fixtures = cases()
    assert len(fixtures) == 17 and len({f['name'] for f in fixtures}) == 17
    for ix, f in enumerate(fixtures):
        assert time.process_time()-START < 15, 'internal CPU cap'
        row = dict(index=ix, name=f['name'], kind=f['kind'], fixture=f, status='started', solves=[])
        D['rows'].append(row); save(); event('case_started', index=ix, name=f['name'])
        def record(entry):
            if entry is not None:
                row['solves'].append(entry); D['solve_attempts'] += 1
            else:
                D['solve_returns'] = sum(s['status'] == 'returned' for rr in D['rows'] for s in rr['solves'])
            save()
        def keep(key, val):
            row[key] = val; save(); return val
        M, mm = a.matrix, a.mul
        x = M(f['x'])
        k = f['kind']
        if k in ('commute', 'common', 'routing', 'pair'):
            h, readout = M(f['hidden']), M(f['readout'])
            y = keep('targets', mm(h, readout))
        if k == 'commute':
            bh = keep('hidden_coefficients', a.ridge(x, h, f['penalty'], record))
            bl = keep('logit_coefficients', a.ridge(x, y, f['penalty'], record))
            pp = keep('predictions', mm(M(f['query']), bl))
            ll = keep('labels', a.labels(pp))
            assert bh == M(f['expected_hidden']) and bl == M(f['expected_logits']) == mm(bh, readout)
            assert pp == M(f['expected_prediction']) and ll == f['expected_labels']
        elif k == 'common':
            yp = keep('shifted_targets', [[v+Q(f['shift'][i]) for v in r] for i, r in enumerate(y)])
            b = keep('base_coefficients', a.ridge(x, y, f['penalty'], record))
            bp = keep('shifted_coefficients', a.ridge(x, yp, f['penalty'], record))
            diff = keep('coefficient_difference', [[u-v for u, v in zip(r, s)] for r, s in zip(bp, b)])
            assert diff == [[Q(v)]*len(y[0]) for v in f['expected_shift']]
            pp, pq = keep('base_predictions', mm(M(f['query']), b)), keep('shifted_predictions', mm(M(f['query']), bp))
            assert a.centered(pp) == a.centered(pq) and a.labels(pp) == a.labels(pq) == f['expected_labels']
        elif k == 'intercept':
            b = keep('coefficients', a.ridge(x, M(f['y']), f['penalty'], record))
            pp = keep('predictions', mm(M(f['query']), b)); keep('labels', a.labels(pp))
            assert b == M(f['expected_coefficients']) and pp == M(f['expected_prediction']) and a.labels(pp) == f['expected_labels']
        elif k == 'routing':
            fits = keep('fits', [])
            for ids in f['groups']:
                bh = a.ridge([x[i] for i in ids], [h[i] for i in ids], f['penalty'], record)
                bl = a.ridge([x[i] for i in ids], [y[i] for i in ids], f['penalty'], record)
                fits.append(dict(hidden=bh, logits=bl)); save()
                assert mm(bh, readout) == bl
            pp = keep('predictions', [mm(M([z]), fits[g]['logits'])[0] for z, g in zip(f['route_queries'], f['routes'])])
            assert pp == M(f['expected_routed'])
        elif k == 'pair':
            b = keep('ordinary', a.ridge(x, y, f['penalty'], record))
            pair = keep('pair', a.pair_fit(x, y, f['penalty'], f['convention'], f['sample_mean'], True, record))
            bc = pair['coefficients']
            keep('predictions', mm(M(f['query']), bc))
            expected = M([['2/3', '1/6', '-5/6'], ['-1/3', '1/3', 0]])
            assert b == M(f['expected_logits']) and bc == expected == a.centered(b)
            assert a.labels(row['predictions']) == f['expected_labels']
            residual = [[u-v for u, v in zip(r, s)] for r, s in zip(mm(x, bc), y)]
            pairs = pair['pairs']
            raw = sum(((r[c]-r[e])**2 for r in residual for c, e in pairs), Q(0))
            pairdata = pair['data_weight']*raw
            centernorm = sum((v*v for r in a.centered(residual) for v in r), Q(0))
            penalty = sum((Q(f['penalty'][j])*v*v for j, r in enumerate(bc) for v in r), Q(0))
            keep('objective', dict(residual=residual, raw_pair_sum=raw, weighted_data=pairdata,
                                   centered_norm=centernorm, unscaled_penalty=penalty,
                                   total=pairdata+pair['penalty_factor']*penalty))
            assert pairdata == pair['alpha']*pair['sample_factor']*centernorm
        elif k == 'unmatched':
            pm = keep('matched', a.pair_fit(x, M(f['y']), f['penalty'], 'unordered_sum', False, True, record))
            pu = keep('unmatched', a.pair_fit(x, M(f['y']), f['penalty'], 'unordered_sum', False, False, record))
            pp = keep('predictions', [mm(M(f['query']), p0['coefficients'])[0] for p0 in (pm, pu)])
            keep('labels', a.labels(pp))
            assert pm['coefficients'] == M(f['expected_matched']) and pu['coefficients'] == M(f['expected_unmatched'])
            assert pp == M(f['expected_predictions']) and a.labels(pp) == f['expected_labels']
        elif k == 'specific':
            ys = M(f['y'])
            bs = keep('coefficients', [])
            for j, pen in enumerate(f['penalties']):
                bs.append(a.ridge(x, [[r[j]] for r in ys], [pen], record)); save()
            direct = keep('direct_sum', a.ridge(x, [[sum(r, Q(0))] for r in ys], [f['sum_penalty']], record))
            vals = keep('values', [mm(M(f['query']), b)[0][0] for b in bs]+[mm(M(f['query']), direct)[0][0]])
            assert vals == [Q(v) for v in f['expected']] and sum(vals[:2]) != vals[2]
        elif k == 'singular':
            caught = None
            try:
                a.ridge(x, M(f['y']), f['penalty'], record)
            except ValueError as exc:
                caught = str(exc)
            keep('caught', caught)
            maps = [M(f[key]) for key in ('fixed_coefficients', 'alternative_coefficients', 'sum_coefficients')]
            training = keep('training_predictions', [mm(x, b) for b in maps])
            queries = keep('query_predictions', [mm(M(f['query']), b)[0] for b in maps])
            assert caught == f['expected_error'] and training[:2] == [M(f['y'])]*2 and training[2] == M([[5]])
            assert queries == M(f['expected_queries'][:2])+[[Q(7)]] and sum(queries[0]) != queries[2][0]
        elif k == 'relu':
            b = keep('coefficients', a.ridge(x, M(f['hidden']), f['penalty'], record))
            pp = keep('linear', mm(mm(M(f['query']), b), M(f['readout'])))
            pc = keep('clipped', mm([[max(Q(0), v) for v in r] for r in mm(M(f['query']), b)], M(f['readout'])))
            ll = keep('labels', [a.labels(pp)[0], a.labels(pc)[0]])
            assert pp == M(f['expected_linear']) and pc == M(f['expected_clipped']) and ll == f['expected_labels']
        else:
            raise AssertionError('unknown fixture')
        row['status'] = 'passed'; save(); event('case_passed', index=ix, name=f['name'])
    assert D['solve_attempts'] == 36 and D['solve_returns'] == 35
    assert sum(s['status'] == 'failed' for r in D['rows'] for s in r['solves']) == 1
    assert time.process_time()-START < 15, 'internal CPU cap'
    event('completed', cases=17, solve_attempts=36, solve_returns=35)
    assert len((OUT/'progress.jsonl').read_text().splitlines()) == 36
    D.update(complete=True, pass_all=True, event_count=36); save()


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if OWNED:
            D.update(complete=False, pass_all=False, failure=dict(type=type(exc).__name__, message=str(exc)))
            save()
        raise
