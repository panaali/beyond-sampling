"""Saved census/category composition only; no determinant evaluations."""
from collections import Counter
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
H4=Path('/workspace/work/recover-small_h4_4/m16/approaches')
A34=Path('/workspace/work/recover-small_h2_34/m16/approaches')
parts={k:json.loads((HERE/(k+'.json')).read_text()) for k in ('baseline','instrumented')}
assert all(v['complete'] for v in parts.values())
a=parts['baseline']['rows'];b=parts['instrumented']['rows'];assert len(a)==len(b)==18
assert [{k:v for k,v in r.items() if k!='spy'} for r in b]==a
assert all(c['passed'] for r in a for c in r.get('reference_checks',[]))
observed=Counter()
for r in b:
    c=r['spy'];observed.update(c);assert c['log_calls']==4*c['success'];assert c['exp_calls']==c['getrf_calls']
    assert all(c[k]==c['getrf_calls'] for k in ('m4','n4','lda4'))
# LU categories from independently reviewed T150 and T157 source envelope.
lu={'floating_arithmetic':56,'absolute_value':40,'floating_compare_or_max':58}
wrapper={'floating_arithmetic':5,'floating_negation':8,'floating_compare_or_max':4,'log':4,'exp':1}
pivot={'pivot_integer_compare':4,'pivot_integer_accumulation':4,'pivot_integer_remainder':1,'pivot_parity_boolean':1,'getrf_info_integer_compare':1}
combined=Counter(lu);combined.update(wrapper);numeric_total=sum(combined.values());assert numeric_total==176
source_paths=[A34/'inverse4_cost_review/DERIVATION.md',A34/'inverse4_cost_review/environment.json',A34/'inverse4_cost_review/dgetrf_single_disassembly.txt',A34/'inverse_numeric_cost/DERIVATION.md',A34/'inverse_numeric_cost/summary.json']
source_pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
compositions={}
for mode,folder,n in [('exact','plackett_cost_review',35098),('recipe','recipe_cost_review',28000)]:
    path=H4/folder/'summary.json';s=json.loads(path.read_text());calls=sum(x['count'] for x in s['opaque_calls'] if x['key'][1]=='np.linalg.det');assert calls==n
    compositions[mode]=dict(calls=calls,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),categories={k:v*calls for k,v in combined.items()},numeric_subset_upper=numeric_total*calls,pivot_source_categories={k:v*calls for k,v in pivot.items()},numeric_plus_pivot_source_subset_upper=(numeric_total+sum(pivot.values()))*calls,parent_visible=s['total_visible_explicit_with_cold_setup'],parent_visible_plus_this_subset=s['total_visible_explicit_with_cold_setup']+(numeric_total+sum(pivot.values()))*calls)
out=dict(complete=True,fixtures=18,actual_matrix_calls=observed['getrf_calls'],observed_spy_totals=dict(observed),lu_numeric_upper=lu,wrapper_numeric_upper=wrapper,pivot_source_upper=pivot,per_det_numeric_upper=numeric_total,per_det_numeric_plus_pivot_source_upper=numeric_total+sum(pivot.values()),compositions=compositions,dependency_pins=source_pins,controls_cpu=sum(p['cpu_seconds'] for p in parts.values()),peak_rss_kib=max(p['peak_rss_kib'] for p in parts.values()),no_trained_replay=True,complete_portable_cost=False,qualification='Subset upper counts: elementary log/exp calls each one named unit, not their internal FLOPs. Loop/index/allocator/copy/interpreter/dispatch and libm internals excluded. This is neither a complete total upper bound nor a total lower bound. LU T157 review status must accompany handoff.')
(HERE/'summary.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({k:out[k] for k in ('complete','fixtures','actual_matrix_calls','per_det_numeric_upper','per_det_numeric_plus_pivot_source_upper','controls_cpu')}))
