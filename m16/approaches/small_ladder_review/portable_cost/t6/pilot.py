"""Identity normalization and fixed64-cone partial integration pilot."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from ..accounting import Audit
from .integration import MassEngine
from .planning import radial_data,tail_bound
HERE=Path(__file__).resolve().parent
def main():
 start=time.process_time();out=dict(identity=[],pilot=[],full_accuracy_estimated=False)
 def save():
  out['cpu_seconds']=time.process_time()-start;(HERE/'pilot_results.json').write_text(json.dumps(out,indent=2)+'\n')
 for n in (8,16):
  audit=Audit();engine=MassEngine(audit);v,e,s,stats=engine.one(np.eye(6),n,1e-10);bound=tail_bound(radial_data(np.eye(6),audit),n,audit)
  assert abs(v-1/64)<=bound+e+s+1e-12
  out['identity'].append(dict(order=n,estimate=v,truth=1/64,outer_bound=bound,inner_bound=e,series_bound=s,cost=audit.report(),stats=stats));save()
 planned=json.loads((HERE/'planning_results.json').read_text());cache=HERE/'projected_correct_rays.npz'
 assert hashlib.sha256(cache.read_bytes()).hexdigest()==planned['projected_cache_sha256']
 data=np.load(cache);rays=data['rays'];projection=data['projection_bounds'];plan=np.load(HERE/'plan_1e-06.npz');orders=plan['orders'];bounds=plan['bounds']
 indices=np.linspace(0,len(rays)-1,64,dtype=int);out['indices']=indices.tolist();out['projected_source_sha256']=planned['projected_cache_sha256']
 audit=Audit();engine=MassEngine(audit);total=0.;inner=0.;series=0.;begin=time.process_time()
 for index in indices:
  try:
   value,error,ser,stats=engine.one(rays[index],int(orders[index]),1e-8/len(rays))
   total+=value;inner+=error;series+=ser
   out['pilot'].append(dict(index=int(index),order=int(orders[index]),mass=value,inner_bound=error,series_bound=ser,stats=stats))
  except Exception as exc:
   out.update(failure_index=int(index),error_type=type(exc).__name__,error=str(exc));break
  out.update(partial_mass=total,partial_outer_bound=float(bounds[indices[:len(out['pilot'])]].sum()),partial_projection_bound=float(projection[indices[:len(out['pilot'])]].sum()),partial_inner_bound=inner,partial_series_bound=series,pilot_cost=audit.report(),pilot_cpu=time.process_time()-begin);save()
  if time.process_time()-start>95:out['error']='pilot CPU cap';break
 out.update(pilot_cost=audit.report(),pilot_cpu=time.process_time()-begin);save()
 print(json.dumps({k:out[k] for k in ('partial_mass','partial_outer_bound','partial_inner_bound','partial_series_bound','pilot_cpu','cpu_seconds','error') if k in out}),flush=True)
 print('completed',len(out['pilot']),'ops',audit.known,'events',dict(audit.events),flush=True)
if __name__=='__main__':main()
