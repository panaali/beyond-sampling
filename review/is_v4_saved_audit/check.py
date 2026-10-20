"""T461: independent saved arithmetic only; never imports estimator code."""
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

PIN = 'a52da0d6a57d29aa2f76daa74c3360222f33aec2'
FREEZE = '91e85991'
PREREG = '5ae28296'
BASE = 'm16/approaches/is_v4/'
manifest = {}


def blob(name, pin=PIN):
    data = subprocess.check_output(['git', 'show', f'{pin}:{BASE}{name}'])
    manifest[f'{pin}:{name}'] = hashlib.sha256(data).hexdigest()
    return data


def obj(name):
    return json.loads(blob(name))


def close(a, b):
    assert math.isclose(a, b, rel_tol=2e-12, abs_tol=1e-30), (a, b)


panel, saved = obj('panel.json'), obj('score4.json')
assert len(panel) == len(saved['rows']) == 15
rows = []
for i, pt in enumerate(panel):
    assert pt == {'model': f'dev3_s{500+i}', 't0': [1.25, 1.75, 2.25][i % 3]}
    e, t = obj(f'est_{i}.json'), obj(f'truth_{i}.json')
    assert blob(f'est_{i}.json', FREEZE) == blob(f'est_{i}.json')
    for x in (e, t):
        assert x['model'] == pt['model'] and x['t0'] == pt['t0'] and x['index'] == i
    close(t['truth'], t['Z'] * t['hits'] / t['n'])
    close(t['se'], t['Z'] * math.sqrt(t['hits']) / t['n'])
    close(e['Z'], t['Z'])
    close(e['cap'], e['Z'] / .2)
    close(e['rel_se'], e['se'] / e['p'])
    flag = e['gate_maxw'] > .05 or (e['n_capped'] < 30 and e['gate_capshare'] > .2) or e['p'] == 0
    assert flag == e['flag']
    delta = e['p'] - t['truth']
    z = delta / math.hypot(e['se'], t['se'])
    pq = t['truth'] * (1-t['truth'])
    f, fd = delta**2 * e['C'] / pq, delta**2 * e['C_draws'] / pq
    r = dict(i=i, ratio=e['p']/t['truth'], z=z, F=f, F_draws=fd,
             raw_ok=t['hits'] >= 100 and abs(z) <= 2,
             honest=t['hits'] >= 100 and (abs(z) <= 2 or flag),
             within2=.5 <= e['p']/t['truth'] <= 2, flag=flag)
    for key in r:
        if isinstance(r[key], bool):
            assert r[key] == saved['rows'][i][key]
        else:
            close(r[key], saved['rows'][i][key])
    r.update(error2=delta**2, pq=pq, C=e['C'], C_draws=e['C_draws'], Z=e['Z'],
             se=e['se'], truth_se=t['se'], hits=t['hits'], extended=e['extended'],
             triple_cardinality_grew=len(e['triples_after']) > len(e['triples_before']),
             new_triple_present=bool(set(e['triples_after']) - set(e['triples_before'])))
    rows.append(r)

mean = statistics.mean
mse, pq = mean(r['error2'] for r in rows), mean(r['pq'] for r in rows)
c, cd = mean(r['C'] for r in rows), mean(r['C_draws'] for r in rows)
summary = dict(n=15, honest=sum(r['honest'] for r in rows), raw_z_le2=sum(r['raw_ok'] for r in rows),
               flags=sum(r['flag'] for r in rows), line_pass=sum(r['honest'] for r in rows) >= 14,
               within2=sum(r['within2'] for r in rows), F_panel=mse*max(c,1000)/pq,
               F_panel_draws=mse*max(cd,1000)/pq, F_le1=sum(r['F'] <= 1 for r in rows),
               beats_pref=sum(r['F'] < r['Z'] for r in rows),
               beats_pref_draws=sum(r['F_draws'] < .19+r['Z'] for r in rows),
               C_mean=c, C_max=max(r['C'] for r in rows), median_ratio=statistics.median(r['ratio'] for r in rows))
assert summary.keys() == saved['summary'].keys()
for k, v in summary.items():
    close(v, saved['summary'][k])
for early, late in ((PREREG, FREEZE), (FREEZE, PIN)):
    subprocess.run(['git', 'merge-base', '--is-ancestor', early, late], check=True)
freeze_paths = subprocess.check_output(['git','ls-tree','-r','--name-only',FREEZE,BASE], text=True).splitlines()
assert not any(Path(p).name.startswith('truth_') and p.endswith('.json') for p in freeze_paths)
for name in ('panel.json','is4.py','run4.py'):
    assert blob(name,PREREG) == blob(name,FREEZE) == blob(name)
extra = dict(min_ratio=min(r['ratio'] for r in rows), max_ratio=max(r['ratio'] for r in rows),
             max_abs_z=max(abs(r['z']) for r in rows),
             truth_hits_range=[min(r['hits'] for r in rows),max(r['hits'] for r in rows)],
             extensions=sum(r['extended'] for r in rows), triple_cardinality_grew=sum(r['triple_cardinality_grew'] for r in rows),
             new_triple_present=sum(r['new_triple_present'] for r in rows),
             mean_truth_variance_over_observed_mse=mean(r['truth_se']**2 for r in rows)/mse,
             F_using_estimator_reported_variance=mean(r['se']**2 for r in rows)*c/pq)
result = dict(status='PASS: saved arithmetic and committed freeze chronology only', source=PIN,
              summary=summary, diagnostics=extra, rows=rows, sha256=manifest)
Path(__file__).with_name('checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],summary=summary,diagnostics=extra),indent=2))
