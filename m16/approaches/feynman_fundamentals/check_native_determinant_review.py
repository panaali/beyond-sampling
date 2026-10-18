"""Independent saved actual determinant dispatch/category audit."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,time,math
start=time.process_time();H=Path(__file__).parent;P=Path('/workspace/work/recover-small_h2_56/m16/approaches/determinant_native_cost');B=json.loads((P/'baseline.json').read_text());I=json.loads((P/'instrumented.json').read_text());S=json.loads((P/'summary.json').read_text());totals={};exact=0
for p,sha in S['dependency_pins'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha
assert len(B['rows'])==len(I['rows'])==18
def determinant(a):
 a=[list(r) for r in a];d=F(1)
 for j in range(4):
  k=next((k for k in range(j,4) if a[k][j]),None)
  if k is None:return F(0)
  if k!=j:a[k],a[j]=a[j],a[k];d=-d
  p=a[j][j];d*=p
  for i in range(j+1,4):
   scale=a[i][j]/p
   for k in range(j+1,4):a[i][k]-=scale*a[j][k]
 return d
for b,i in zip(B['rows'],I['rows']):
 assert b=={k:v for k,v in i.items() if k!='spy'}
 n=2 if b['name']=='batch' else 1;s=i['spy']
 assert all(s[k]==n for k in ('getrf_calls','m4','n4','lda4','exp_calls')) and s['log_calls']==4*s['success'];assert s['success']+s['singular']==n and not s['invalid']
 for k,v in s.items():totals[k]=totals.get(k,0)+v
 if 'exact_determinants' in b:
  values=[F(float.fromhex(v)) for v in b['input_hex']]
  for j,q in enumerate(b['exact_determinants']):
   m=[values[16*j+4*k:16*j+4*k+4] for k in range(4)];assert determinant(m)==F(q);exact+=1
assert totals==S['observed_spy_totals'] and totals['getrf_calls']==19
assert S['lu_numeric_upper']==dict(floating_arithmetic=56,absolute_value=40,floating_compare_or_max=58)
assert sum(S['lu_numeric_upper'].values())+sum(S['wrapper_numeric_upper'].values())==176
assert sum(S['pivot_source_upper'].values())==11
for label,n,visible in [('exact',35098,553400768),('recipe',28000,455976715)]:
 r=S['compositions'][label];assert r['calls']==n and r['numeric_subset_upper']==176*n and r['numeric_plus_pivot_source_subset_upper']==187*n and r['parent_visible_plus_this_subset']==visible+187*n
 assert sum(r['categories'].values())==176*n and sum(r['pivot_source_categories'].values())==11*n
out=dict(bitwise_baseline_parity_fixtures=18,actual_matrix_calls=19,independent_exact_gaussian_elimination_checks=exact,successful_log_calls=totals['log_calls'],exp_calls=totals['exp_calls'],numeric_category_upper=176,numeric_plus_selected_source_integer_upper=187,review_cpu=time.process_time()-start,scope='Actual GETRF/log-exp dispatch and explicit category subtotals; libm internals, source loop/index and general control/memory/runtime excluded. Neither complete cost nor determinant-roundoff certificate.')
(H/'native_determinant_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
