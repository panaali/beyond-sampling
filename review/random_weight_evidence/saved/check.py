"""Frozen scalar-only reconciliation. No author imports, arrays or estimator calls."""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time
import traceback
from fractions import Fraction as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AUTHOR = ROOT / 'frozen/author/m16/approaches/random_control'
OUT = HERE / 'output'
MODELS = ['target'] + [f'frob{i}' for i in range(5)] + [f'row{i}' for i in range(5)] + ['perm0', 'flip0']
DELTAS = ['1e-02', '1e-04', '1e-06']
START = time.process_time()
RESULT = {'task': 'T248 scalar evidence reconciliation', 'complete': False, 'pass_all': False,
          'rows': [], 'groups': {}, 'author_calls': 0, 'array_decodes': 0}


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump():
    tmp = OUT / 'results.tmp'
    tmp.write_text(json.dumps(RESULT, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, OUT / 'results.json')


def event(value):
    assert time.process_time() - START < 16
    with (OUT / 'progress.jsonl').open('a') as f:
        f.write(json.dumps(value, allow_nan=False) + '\n')


def exact(q):
    q = F(q)
    return {'numerator': q.numerator, 'denominator': q.denominator, 'display': float(q)}


def median(xs):
    a = sorted(map(F, xs)); n = len(a)
    return a[n // 2] if n % 2 else (a[n // 2 - 1] + a[n // 2]) / 2


def finite(x):
    assert type(x) in (int, float) and math.isfinite(x)


def validate_prerequisites():
    protocol = read(HERE / 'protocol.json'); gate = read(HERE / 'gate.json')
    assert gate['status'] == 'ready'
    for key in ('source_review_publication', 'source_gate_publication', 'execution_scope_publication'):
        assert isinstance(gate[key], str) and len(gate[key]) >= 8
        int(gate[key], 16)
    expected = [HERE / 'check.py', HERE / 'protocol.json', ROOT / 'manifest.json', ROOT / 'SOURCE_FINDINGS.md', ROOT / 'README.md',
                Path(protocol['source_review']), Path(protocol['source_gate'])]
    pins = gate['pins']; assert all(str(p) in pins for p in expected)
    for p, h in pins.items():
        assert sha(p) == h, p
    review = read(protocol['source_review'])
    assert review['complete'] is True and review['pass_all'] is True
    clearance = read(protocol['source_gate'])
    assert clearance['status'] == 'passed' and clearance['execution_authorized'] is True
    for obj, required in [(review, [ROOT / 'manifest.json', ROOT / 'SOURCE_FINDINGS.md', ROOT / 'README.md']),
                          (clearance, [HERE / 'check.py', HERE / 'protocol.json', ROOT / 'manifest.json', Path(protocol['source_review'])])]:
        assert all(str(p) in obj['pins'] for p in required)
        for p, h in obj['pins'].items():
            assert sha(p) == h, p
    manifest = read(ROOT / 'manifest.json'); assert len(manifest['payloads']) == 50
    for row in manifest['payloads']:
        p = ROOT / row['snapshot']; assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256']
    assert 0 < resource.getrlimit(resource.RLIMIT_CPU)[0] <= 20
    assert 0 < resource.getrlimit(resource.RLIMIT_CPU)[1] <= 20
    for cap in resource.getrlimit(resource.RLIMIT_AS):
        assert 0 < cap <= 2 * 1024**3
    RESULT['pins'] = pins
    RESULT['resource_limits'] = {'CPU': resource.getrlimit(resource.RLIMIT_CPU), 'AS': resource.getrlimit(resource.RLIMIT_AS)}


def main():
    validate_prerequisites()
    event({'status': 'started'})
    expected_files = {f'{m}_k{k}.json' for m in MODELS for k in (2, 3, 4)}
    assert {p.name for p in (AUTHOR / 'out').iterdir()} == expected_files
    raw = {}
    for m in MODELS:
        for k in (2, 3, 4):
            row = {'model': m, 'k': k, 'status': 'started'}; RESULT['rows'].append(row)
            dump(); event({'model': m, 'k': k, 'status': 'started'})
            r = read(AUTHOR / 'out' / f'{m}_k{k}.json'); row['raw'] = r; dump()
            assert r['model'] == m and r['k'] == k
            for key in ('cells', 'regions', 'ops_geometry'):
                assert type(r[key]) is int and r[key] > 0
            assert r['regions'] <= r['cells']
            for key in ('cpu', 'cpu_total'):
                finite(r[key]); assert r[key] >= 0
            assert r['cpu_total'] >= r['cpu']
            if k == 2:
                for key in ('acc_1e6', 'active_frac_T10'):
                    finite(r[key]); assert 0 <= r[key] <= 1
                n = round(r['acc_1e6'] * 1_000_000)
                row['reported_mean_compatible_hits'] = n
                assert n / 1_000_000 == r['acc_1e6']
            if k == 4:
                assert set(r['N']) == set(DELTAS)
                ns = [r['N'][d] for d in DELTAS]
                assert all(type(n) is int and 1 <= n <= r['regions'] for n in ns)
                assert ns == sorted(ns)
                finite(r['mass_sum_minus_1']); finite(r['inv_simpson'])
                assert r['mass_sum_minus_1'] > -1 and r['inv_simpson'] > 0
            row['status'] = 'complete'; raw[m, k] = r; dump()
            event({'model': m, 'k': k, 'status': 'complete'})
    summary = read(AUTHOR / 'summary.json'); assert set(summary) == set(MODELS)
    for m in MODELS:
        a, b, c = [raw[m, k] for k in (2, 3, 4)]
        expected = {'regions': [a['regions'], b['regions'], c['regions']],
                    'growth': [float(F(b['regions'], a['regions'])), float(F(c['regions'], b['regions']))],
                    'acc': a['acc_1e6'], 'active': a['active_frac_T10'], 'N': c['N'],
                    'inv_simpson': c['inv_simpson'], 'msum': c['mass_sum_minus_1']}
        assert summary[m] == expected, m
    RESULT['summary_models_checked'] = len(summary)
    log_seen = set()
    for line in (AUTHOR / 'drive.log').read_text().splitlines():
        m, ktext, body = line.split(' ', 2); k = int(ktext); key = m, k
        assert key in raw and key not in log_seen
        assert ast.literal_eval(body) == {x: v for x, v in raw[key].items() if x != 'model'}
        log_seen.add(key)
    RESULT['log_coverage'] = {'matched': len(log_seen), 'missing': [[m, k] for m in MODELS for k in (2, 3, 4) if (m, k) not in log_seen],
                              'scope': 'Missing log rows are retained, not fabricated or treated as absent results.'}
    seen = set()
    for line in (AUTHOR / 'README.md').read_text().splitlines():
        cells = [x.strip() for x in line.split('|')]
        if len(cells) != 9 or cells[1] not in MODELS:
            continue
        m = cells[1]; assert m not in seen; seen.add(m); s = summary[m]
        expected = ['', m, ' / '.join(f'{x:,}' for x in s['regions']),
                    ', '.join(f'x{x:.1f}' for x in s['growth']),
                    ' / '.join(f"{s['N'][d]:,}" for d in DELTAS), f"{s['inv_simpson']:.0f}",
                    f"{s['acc']:.4f}", f"{s['active']:.2f}", '']
        assert cells == expected, {'model': m, 'actual': cells, 'expected': expected}
    assert seen == set(MODELS)
    RESULT['rounded_table_rows_checked'] = len(seen)
    target_growth = F(raw['target', 4]['regions'], raw['target', 3]['regions'])
    groups = {'frob5': MODELS[1:6], 'row5': MODELS[6:11], 'random10': MODELS[1:11], 'nontarget12_including_controls': MODELS[1:]}
    for group, names in groups.items():
        growth = [F(raw[m, 4]['regions'], raw[m, 3]['regions']) for m in names]
        med = median(growth); relative = med / target_growth
        nmed = median([raw[m, 4]['N']['1e-04'] for m in names]) / raw['target', 4]['N']['1e-04']
        strong = relative >= 2 and nmed >= 3
        within = F(7, 10) <= relative <= F(13, 10)
        verdict = 'strong' if strong else ('within30percent' if within else 'moderate')
        ng = {}
        for d in DELTAS:
            ratios = [F(raw[m, 4]['N'][d], raw['target', 4]['N'][d]) for m in names]
            ng[d] = {'minimum': exact(min(ratios)), 'median': exact(median(ratios)), 'maximum': exact(max(ratios))}
        RESULT['groups'][group] = {'models': names, 'median_growth_k3': exact(median([F(raw[m, 3]['regions'], raw[m, 2]['regions']) for m in names])),
                                  'median_growth_k4': exact(med), 'relative_growth_k4': exact(relative),
                                  'reported_N_ratio': ng, 'descriptive_category': verdict,
                                  'preregistered_decision_group': group == 'random10'}
    RESULT['reported_cpu_total_sum'] = exact(sum((F(raw[m, k]['cpu_total']) for m in MODELS for k in (2, 3, 4)), F()))
    RESULT['max_abs_reported_mass_sum_minus_1'] = exact(max(abs(F(raw[m, 4]['mass_sum_minus_1'])) for m in MODELS))
    RESULT['qualifications'] = ['Aggregate consistency only: masses, threshold minimality, geometry and realized model identity unverified.',
                                'Per-ensemble and all-control ranges are separate; only random10 is the preregistered decision group.',
                                'Descriptive thresholds are not equivalence tests or universal growth laws; no accuracy confidence interval.',
                                'No new CI, sampling, mass integration, source-native cost or target experiment.']
    RESULT['complete'] = RESULT['pass_all'] = True
    RESULT['cpu_seconds_main'] = time.process_time() - START
    event({'status': 'complete'}); dump()


if __name__ == '__main__':
    assert not OUT.exists() or not any(OUT.iterdir()), 'refuse existing output'
    OUT.mkdir(exist_ok=True)
    try:
        main()
    except BaseException:
        RESULT['error'] = traceback.format_exc(); RESULT['cpu_seconds_main'] = time.process_time() - START
        dump(); (OUT / 'failure.txt').write_text(RESULT['error'])
        raise
