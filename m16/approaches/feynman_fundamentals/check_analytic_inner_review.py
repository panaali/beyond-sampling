"""Frozen-order saved-array and explicit-ledger audit of exact-inner rescue."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import json,time,hashlib,math
import numpy as np
start=time.process_time();here=Path(__file__).parent;root=Path('/workspace/work/recover-small_h2_34/m16/approaches');src=root/'analytic_inner_mass';truth=np.load(root/'radial_t4_reference/combined_values.npz');out=[]
for tag in ['per_cone','aggregate']:
 d=json.loads((src/f'{tag}.json').read_text());old=json.loads((src/f'{tag}_initial.json').read_text());base=json.loads((root/'positive_mass_budget'/f'{tag}.json').read_text());v=np.load(src/f'{tag}_values.npz');ledger=Counter();modes=Counter();additional_a=additional_c=0
 assert len(d['rows'])==len(base['rows'])==6347 and d['complete'];assert hashlib.sha256((root/'positive_mass_budget'/f'{tag}.json').read_bytes()).hexdigest()==d['base_sha256']
 for i,(r,o,b) in enumerate(zip(d['rows'],old['rows'],base['rows'])):
  assert r['index']==i and r['mode']==b['mode'];assert r.get('value')==o.get('value');assert r.get('old_value')==b.get('value');assert r.get('old_bound')==b.get('bound')
  if 'value' in r:assert r['value']==v['values'][i] and b['bound']==v['bounds'][i]
  if r['mode']=='quadrature':
   n=r['order'];assert n==b['order'];m=n*n;assert r['inner_modes']=={'analytic':m};a=m;c=(21 if o['kernel_counts'].get('sin',0) else 18)*m-5;expected=Counter(o['kernel_counts']);expected['arithmetic']+=a;expected['comparison']+=c;assert expected==r['kernel_counts'];additional_a+=a;additional_c+=c;assert r['series_error_bound']==v['series_bounds'][i]
  ledger.update(r.get('kernel_counts',{}));modes.update(r.get('inner_modes',{}))
 assert dict(ledger)==d['kernel_counts'] and dict(modes)==d['inner_modes'];assert d['named_total_including_comparisons']==old['named_total_including_comparisons']+additional_a+additional_c
 total=sum(ledger.values())+sum(d['cold_node_plan_counts'].values())+d['inherited_setup_selection_count']+d['inherited_ordinary_constant_arithmetic'];assert total==d['named_total_including_comparisons']
 mask=np.isfinite(v['values']);assert int(mask.sum())==d['covered'];assert np.flatnonzero(~mask).tolist()==d['unresolved_indices'];err=sum((F(float(x))-F(float(y)) for x,y in zip(v['values'][mask],truth['values'][mask])),F(0));assert abs(float(err)-d['covered_error'])<1e-19
 out.append(dict(budget=tag,covered=d['covered'],unresolved=len(d['unresolved_indices']),named_total=total,old_named_total=d['old_full_named_total'],reduction=1-total/d['old_full_named_total'],outer_analytic_nodes=modes['analytic'],covered_mass_error=float(err),series_bound=math.fsum(v['series_bounds'][mask]),extra_arithmetic=additional_a,extra_comparisons=additional_c))
result=dict(rows=out,review_cpu=time.process_time()-start,scope='Saved values, frozen order/bound/coverage parity and corrected explicit ledgers. Real-arithmetic polynomial exactness transfer plus separate series remainder; no floating geometry/analytic-roundoff or full implementation-cost certificate.')
(here/'analytic_inner_review_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
