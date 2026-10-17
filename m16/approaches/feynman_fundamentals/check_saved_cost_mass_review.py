"""Raw saved inventories only: no estimator or integration reruns."""
from pathlib import Path
from fractions import Fraction as F
import json,math,time
import numpy as np
start=time.process_time();base=Path('/workspace/work');p=base/'recover-small_h2_78/m16/approaches/fibre_cost_review';d=json.loads((p/'arithmetic_panel_results.json').read_text());sampler=json.loads((p/'sampler_native_results.json').read_text());ratios=[]
for r in d['rows']:
 l=r['audit']['ledger'];counts={}
 for site,v in l['sites'].items():
  for k,n in v.items():counts[k]=counts.get(k,0)+n
 assert counts==l['totals'];integer=sum(n for k,n in counts.items() if k.startswith('integer_control'));total=sum(counts.values())
 assert integer==l['integer_control_total'] and total-integer==l['named_numeric_total'];s=sampler[r['name']];assert s['named_numeric_total']==111 and s['integer_control_total']==1
 assert all(r['parity'].values())
 ratio=F(total*72,(r['audit']['n']+r['audit']['x'])*112);assert abs(float(ratio)-r['named_plus_integer_vs_historical_ratio'])<=2*math.ulp(float(ratio))
 ratios.append({'name':r['name'],'explicit_expression_count':total,'ratio_to_historical':float(ratio)})
p=base/'recover-small_h2_34/m16/approaches';r=json.loads((p/'positive_t4_corpus/results.json').read_text());v=np.load(p/'positive_t4_corpus/positive_values.npz');source=np.load(p/'cone_mass4_review/quadratic_values.npz')['values'];guard=np.load(p/'cone_mass4_review/quadratic_rotation_values.npz')['values'];finite=np.isfinite(v['values']);assert sum(finite)==5859 and sum(~finite)==488
sumf=lambda a:sum((F(float(x)) for x in a),F(0))
mass=sumf(v['values'][finite]);bound=sumf(v['bounds'][finite]);sd=sumf(source[finite])-mass;gd=sumf(guard[finite])-mass;res=sumf(source[~finite]);diagnostic=1-mass-res
for actual,key in [(mass,'reference_sum'),(bound,'integration_only_bound_sum'),(res,'uncovered_source_sum')]:assert abs(float(actual)-r[key])<1e-16
mp=json.loads((p/'positive_t4_corpus/mp_controls.json').read_text());assert len(mp['rows'])==3
for row in mp['rows']:
 assert row['positive_reference']==v['values'][row['index']]
 a,b=[F(q['decimal']) for q in row['precisions']];assert abs(a-b)<F('1e-45')
 assert abs(F(float(row['positive_reference']))-b)<F('4e-20')
out={'explicit_cost_inventory':ratios,'positive_covered_mass':float(mass),'integration_only_bound':float(bound),'source_covered_minus_reference':float(sd),'guard_covered_minus_reference':float(gd),'diagnostic_accuracy_with_unverified_residual':float(diagnostic),'mp_checked_positive_discrepancies':3,'cpu_seconds':time.process_time()-start,'scope':'explicit source-expression convention excludes implicit control and memory; positive mass residual unverified'}
Path(__file__).with_name('saved_cost_mass_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
