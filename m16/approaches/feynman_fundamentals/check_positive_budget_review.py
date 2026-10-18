"""Independent call-ledger recount and saved covered-reference sums."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import json,time
import numpy as np
start=time.process_time();p=Path('/workspace/work/recover-small_h2_34/m16/approaches');reference=np.load(p/'radial_t4_reference/combined_values.npz')['values'];out=[]
for name in('per_cone','aggregate'):
 d=json.loads((p/f'positive_mass_budget/{name}.json').read_text());v=np.load(p/f'positive_mass_budget/{name}_values.npz');counts=Counter();kernel=0;constants=0;modes=Counter();orders=set()
 for r in d['rows']:
  counts.update(r['setup_selection_counts']);modes[r['mode']]+=1;constants+=2*('interval' in r)+2*bool(r['trials'])+2*(r['mode']=='quadrature');kernel+=r['kernel_arithmetic']
  if r['mode']=='quadrature':n=r['order'];orders.add(n);assert r['kernel_arithmetic']==39*n**3+1 and r['trials'][-1]['bound']<=d['tolerance']
  elif r['mode']=='interval':assert r['bound']<=d['tolerance'] and r['kernel_arithmetic']==0
  else:assert 'value' not in r
 cold=sum(18*n**3+3*n for n in orders);total=sum(counts.values())+kernel+cold+constants
 assert dict(counts)==d['counts'] and constants==d['ordinary_pi_constant_arithmetic'] and kernel==d['kernel_arithmetic'] and cold==d['cold_node_plan_arithmetic'] and total==d['named_total_including_comparisons'] and dict(modes)==d['modes']
 ok=np.isfinite(v['values']);error=sum((F(float(a))-F(float(b)) for a,b in zip(v['values'][ok],reference[ok])),F(0));bound=sum((F(float(x)) for x in v['bounds'][ok]),F(0));assert abs(float(error)-d['covered_error'])<1e-17 and abs(float(bound)-d['bound_sum'])<1e-18
 out.append({'budget':name,'modes':dict(modes),'named_operations_including_cold_plan':total,'pi_constant_operations':constants,'covered_probability_error':float(error),'covered_integration_bound':float(bound),'covered':int(sum(ok))})
result={'rows':out,'cpu_seconds':time.process_time()-start,'scope':'named arithmetic subset with explicit pi correction; unresolved aggregate32 retained; no Legendre/full float certificate'}
Path(__file__).with_name('positive_budget_review_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
