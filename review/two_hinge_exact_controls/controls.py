"""Once-only frozen synthetic checks. Run only after published independent gate."""
import hashlib
import json
import resource
import sys
import time
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def encode(x):
    if isinstance(x, F):
        return [str(x.numerator), str(x.denominator)]
    if isinstance(x, dict):
        return {k: encode(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [encode(v) for v in x]
    return x


def main():
    start = time.process_time()
    out = HERE / 'output'
    out.mkdir(exist_ok=True)
    result = out / 'results.json'
    progress = out / 'progress.jsonl'
    if result.exists() or progress.exists():
        raise RuntimeError('refuse existing result/progress; preserve prior attempts')
    records = []
    report = {'complete': False, 'pass_all': False, 'rows': records}

    def save():
        report['cpu_seconds'] = time.process_time() - start
        result.write_text(json.dumps(encode(report), indent=2) + '\n')

    def event(kind, **kw):
        with progress.open('a') as f:
            f.write(json.dumps(encode({'event': kind, **kw})) + '\n')
        if time.process_time() - start > 17:
            raise RuntimeError('17 CPU-second main-scope cap')

    try:
        event('attempt')
        limits = {'cpu': resource.getrlimit(resource.RLIMIT_CPU),
                  'address_space': resource.getrlimit(resource.RLIMIT_AS)}
        report['runtime'] = {'executable': sys.executable, 'prefix': sys.prefix, 'limits': limits}
        save()
        assert all(0 < x <= 20 for x in limits['cpu'])
        assert all(0 < x <= 2 * 1024**3 for x in limits['address_space'])
        p = json.loads((HERE / 'protocol.json').read_text())
        g = json.loads((HERE / 'gate.json').read_text())
        assert g['execution_allowed'] is True and g['published_commit'] != 'pending'
        assert sys.executable == p['python']
        for pins in (p['pins'], g['pins']):
            for path, sha in pins.items():
                assert digest(path) == sha, path
        assert g['protocol_sha256'] == digest(HERE / 'protocol.json')
        review = json.loads(Path(g['review_path']).read_text())
        assert digest(g['review_path']) == g['review_sha256']
        assert review['complete'] is True and review['pass_all'] is True
        report['protocol_sha256'] = digest(HERE / 'protocol.json')
        report['gate_sha256'] = digest(HERE / 'gate.json')
        from primitive import point, conditional, interval_point, scale
        fixtures = json.loads((HERE / 'fixtures.json').read_text())
        assert len(fixtures) == 12
        report['calls'] = {'point': 0, 'interval_point': 0, 'scale': 0, 'guard_attempts': 0}
        for case in fixtures:
            event('case_attempt', name=case['name'])
            a, b, c, d, di, dj, k = map(F, case['parameters'])
            xs = list(map(F, case['x']))
            assert len(xs) == 5
            row = {'name': case['name'], 'parameters': [a,b,c,d,di,dj,k], 'points': [], 'passed': False}
            records.append(row)
            row['gauge'] = scale(a,b,c,d,di,dj,k)
            report['calls']['scale'] += 1
            save()
            for x in xs:
                ap, bp = a*x+c, b*x+d
                original = point(ap,bp)
                transformed = point(di*ap,dj*bp)
                q, ir = interval_point(a,b,c,d,x)
                report['calls']['point'] += 2
                report['calls']['interval_point'] += 1
                rec = {'x': x, 'a': ap, 'b': bp, 'original': original,
                       'scaled': transformed, 'interval': q, 'interval_residual': ir}
                row['points'].append(rec)
                save()  # retain every returned value before independent assertions
                reference = max(bp,F(0)) - (bp if ap < 0 else F(0))
                assert original['residual'] == reference >= 0
                assert original['surrogate'] + reference == max(bp,F(0))
                assert ir == reference
                assert transformed['residual'] == dj*reference
                assert (k/dj)*transformed['residual'] == k*reference
            q = row['gauge']['original']
            # Separate endpoint polynomial integration, no author area expression.
            v, lo, hi = q['v'], q['lower'], q['upper']
            signed_integral = (hi*hi-lo*lo)/2 - v*(hi-lo)
            row['independent_area'] = abs(b*signed_integral)
            save()
            assert row['independent_area'] == q['area']
            assert row['gauge']['weighted_area'] == row['gauge']['transformed_weighted_area']
            assert q['u'] == row['gauge']['transformed']['u']
            assert q['v'] == row['gauge']['transformed']['v']
            row['passed'] = True
            save()
            event('case_result', name=case['name'], passed=True)
        guards = [
            ('bool', TypeError, lambda: point(True,F(1))),
            ('float', TypeError, lambda: point(1.0,F(1))),
            ('nan', TypeError, lambda: point(float('nan'),F(1))),
            ('zero_alpha', ValueError, lambda: conditional(0,-1,0,0)),
            ('zero_beta', ValueError, lambda: conditional(1,0,0,0)),
            ('same_sign', ValueError, lambda: conditional(1,1,0,0)),
            ('zero_gauge', ValueError, lambda: scale(1,-1,0,0,0,1,1)),
            ('negative_margin', ValueError, lambda: scale(1,-1,0,0,1,1,-1))]
        report['guards'] = []
        for name, expected, call in guards:
            event('guard_attempt', name=name)
            report['calls']['guard_attempts'] += 1
            try:
                call()
            except expected as exc:
                rec = {'name': name, 'passed': True, 'exception': type(exc).__name__}
            else:
                rec = {'name': name, 'passed': False, 'exception': None}
            report['guards'].append(rec)
            save()
            assert rec['passed']
            event('guard_result', **rec)
        assert len(records) == 12 and len(report['guards']) == 8
        assert report['calls'] == {'point':120,'interval_point':60,'scale':12,'guard_attempts':8}
        report['complete'] = report['pass_all'] = True
        save()
        event('complete')
    except BaseException as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        save()
        with progress.open('a') as f:
            f.write(json.dumps({'event':'fatal','type':type(exc).__name__,'message':str(exc)})+'\n')
        raise


if __name__ == '__main__':
    main()
