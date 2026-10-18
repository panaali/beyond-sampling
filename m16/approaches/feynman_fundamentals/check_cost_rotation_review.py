"""Independent saved-ledger arithmetic and exact rotation geometry; no estimator reruns."""
from pathlib import Path
from fractions import Fraction as F
import ast,json,time
import numpy as np
start=time.process_time();base=Path('/workspace/work')
fibre=base/'recover-small_h2_78/m16/approaches/fibre_cost_review'
d=json.loads((fibre/'panel_results.json').read_text());rows=[]
for row in d['rows']:
 a=row['audit'];sites=a['ledger']['sites'];tot=a['ledger']['totals']
 for key in tot:assert sum(v[key] for v in sites.values())==tot[key]
 for key,v in sites.items():
  assert v['mul']==v['dense_mac']
  if 'A * g' in key:assert v['mask_mul']==12*v['calls']
  elif 'Wh @ Ag' in key:assert (v['mul'],v['add'])==(48*v['calls'],36*v['calls'])
  elif 'Wo @ Ag' in key:assert (v['mul'],v['add'])==(36*v['calls'],27*v['calls'])
  elif 'A @ x' in key:assert (v['mul'],v['add'])==(12*v['calls'],8*v['calls'])
  else:assert 3*v['add']==2*v['mul']
 adjusted=a['n']+a['x']-tot['historical_mac']+tot['mul']+tot['add']+tot['mask_mul']
 assert adjusted==a['partial_scalar_adjustment']
 sample=row['sampling_scalar'];assert (sample['mul'],sample['add'],sample['comparison'],sample['total'])==(56,41,15,112)
 assert all(row['parity'].values())
 ratio=F(adjusted*72,(a['n']+a['x'])*112)
 assert abs(float(ratio)-row['partial_vs_historical_budget_ratio'])<3e-16
 rows.append({'name':row['name'],'partial_count':adjusted,'partial_budget_ratio':float(ratio)})
rot=base/'recover-small_h2_34/m16/approaches/cone_mass4_review';r=json.loads((rot/'rotation_results.json').read_text())['rows'][0]
R=np.load(rot.parent/'class_region_t4/nearest_exact_mass_calls.npz')['rays'][r['index']]
c,s=F(r['c']),F(r['s']);assert c*c+s*s==1;i,j=r['pair'];Y=[]
for ray in R:
 z=list(map(lambda v:F(float(v)),ray));a,b=z[i],z[j];z[i]=c*a-s*b;z[j]=s*a+c*b;Y.append(z)
assert all(all(z[q]>=0 for z in Y) or all(z[q]<=0 for z in Y) for q in range(4))
# Exact common orthant is established on stored dyadic rays, independently of floating rotation.
out={'fibre_rows':rows,'fibre_scope':'named kernels and sampler only; remaining scalar ledger open','rotation':{'index':r['index'],'exact_rational_orthogonality':True,'exact_stored_ray_common_orthant':True,'mass_arithmetic_certified':False},'cpu_seconds':time.process_time()-start}
Path(__file__).with_name('cost_rotation_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
