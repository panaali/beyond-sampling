import json,time,resource,hashlib
from pathlib import Path
import numpy as np
from geometry import Geometry
from models import load,noisy
from allocation import allocate,ORDERS
from exact_tail import table
BASE=Path(__file__).parent;start=time.process_time();out={'models':[],'target':1e-5,'omission_reserve':2e-6,'inner_reserve':1e-7}
# Reference integration timing is a cost observation only, never numerical advice.
reference=json.loads((BASE/'cycle13.json').read_text());seconds_per_call=sum(c['cpu_seconds'] for c in reference['cycles'])/sum(c['primitive_counts']['triangles'] for c in reference['chunks'])
def save():
 out['cpu_seconds']=time.process_time()-start;out['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 out['shared_loading_serialization_and_other_cpu_seconds']=max(0.,out['cpu_seconds']-sum(r.get('cold_setup_cpu_seconds',r.get('cpu_seconds',0.)) for r in out['models']))
 for row in out['models']:
  if 'projected_total_cpu_seconds' in row:row['projected_total_charging_all_shared_overhead_cpu_seconds']=row['projected_total_cpu_seconds']+out['shared_loading_serialization_and_other_cpu_seconds']
 (BASE/'cycle16.json').write_text(json.dumps(out,indent=2)+'\n')
base,sha=load(6)
for sigma in (0.,.0025,.02):
 s=time.process_time();model=base if sigma==0 else noisy(base,sigma);g=Geometry(30);row=dict(sigma=sigma,checkpoint_sha256=sha,weights_sha256=hashlib.sha256(b''.join(w.tobytes() for w in model)).hexdigest())
 try:
  cells=g.gates(model);pieces=g.correct(model,cells);rays=g.simplices(pieces);row.update(geometry_cpu_seconds=time.process_time()-s,geometry_counts=g.counts)
  t=time.process_time();errors,upper=table(rays,ORDERS);rank=np.argsort(upper,kind='stable');cumulative=np.cumsum(upper[rank]);reserve=2e-6-g.counts['omitted_mass_upper_bound'];count=int(np.searchsorted(cumulative,reserve,side='right'));omit=rank[:count];mask=np.ones(len(rays),bool);mask[omit]=False;omitted=float(upper[omit].sum());outer=1e-5-omitted-g.counts['omitted_mass_upper_bound']-1e-7
  row.update(original_simplices=len(rays),omitted_simplices=count,retained_simplices=int(mask.sum()),pruned_mass_bound=omitted,table_and_pruning_cpu_seconds=time.process_time()-t,table_determinants=len(rays),table_gram_arithmetic_ops=len(rays)*396,table_supporting_arithmetic_ops=len(rays)*114,bound_entries=errors.size)
  orders,bounds,allocation=allocate(errors[mask],outer,3);row['allocation']=allocation
  row['total_conditional_planned_bound']=float(bounds.sum())+omitted+g.counts['omitted_mass_upper_bound']+1e-7
  row['cold_setup_cpu_seconds']=time.process_time()-s;row['projected_integration_cpu_seconds']=allocation['predicted_nodes']*seconds_per_call;row['projected_total_cpu_seconds']=row['cold_setup_cpu_seconds']+row['projected_integration_cpu_seconds'];row['projection_seconds_per_call']=seconds_per_call;row['total_cost_audited']=False
  name='cycle16_sigma'+str(sigma);np.savez_compressed(BASE/(name+'_rays.npz'),rays=rays[mask]);np.savez_compressed(BASE/(name+'_plan.npz'),orders=orders,bounds=bounds)
 except Exception as exc:row.update(failure=repr(exc),cpu_seconds=time.process_time()-s,geometry_counts=g.counts)
 out['models'].append(row);save();print(row,flush=True)
 if time.process_time()-start>115:raise TimeoutError('robustness planning CPU guard')
save()
