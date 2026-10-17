"""Recompute summaries from every saved exact determinant and interval."""
from fractions import Fraction as F
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
parts=[json.loads((HERE/f'corpus_{i}.json').read_text()) for i in range(4)]
assert all(p['complete'] for p in parts)
rows=sum([p['rows'] for p in parts],[])
assert [r['index'] for r in rows]==list(range(6347))
for r in rows:
    exact=F(r['exact_determinant']);interval=r['interval']
    assert interval['contains_exact'] and F(interval['lo'])<=exact<=F(interval['hi'])
    assert interval['lo']>0 or interval['hi']<0
    for name in ('new47','old95'):
        error=F(r['point'][name]['value'])-exact
        assert float(error/exact)==r['point'][name]['relative_error']
        assert float(abs(error))==r['point'][name]['absolute_error']
better=sum(r['point']['new47']['absolute_error']<r['point']['old95']['absolute_error'] for r in rows)
worse=sum(r['point']['new47']['absolute_error']>r['point']['old95']['absolute_error'] for r in rows)
control=json.loads((HERE/'controls.json').read_text())
summary=dict(cases=len(rows),enclosed_nonzero=len(rows),point47_better=better,point47_worse=worse,
             point_equal=len(rows)-better-worse,point_arithmetic_saved_per_matrix=48,
             point_arithmetic_saved_corpus=48*len(rows),interval_counts_per_matrix=rows[0]['interval']['counts'],
             maximum_relative_interval_width=max(r['interval']['relative_width'] for r in rows),
             corpus_cpu=sum(p['cpu_seconds_including_imports'] for p in parts),
             controls_cpu=control['cpu_seconds_including_imports'],peak_rss_kib=max(p['max_rss_kib'] for p in parts),
             exact_reference_component_cpu_separately_unmeasured=True,
             source_hashes=parts[0]['source_hashes'],point={})
for name in ('new47','old95'):
    worst=max(rows,key=lambda r:abs(r['point'][name]['relative_error']))
    summary['point'][name]=dict(worst_relative_index=worst['index'],maximum_absolute_relative_error=abs(worst['point'][name]['relative_error']),
                               maximum_absolute_error=max(r['point'][name]['absolute_error'] for r in rows),
                               counts=rows[0]['point'][name]['counts'])
assert all(r['interval']['counts']==summary['interval_counts_per_matrix'] for r in rows)
(HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
