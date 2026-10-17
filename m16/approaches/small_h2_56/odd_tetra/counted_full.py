import hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
from counted import Audit,enumerate_correct,volume
BASE=Path(__file__).parent
sys.path.insert(0,str(BASE.parent))
from models import load,noisy
sys.path.pop(0)
start=time.process_time();base,sha=load(5)
out={'checkpoint_sha256':sha,'checkpoint_loading_cpu_seconds':time.process_time()-start,'models':[]}
references={0.:.27923737131514703,.0025:.31063601436566657,.02:.32506108244184867}
reference_errors={0.:2.98e-9,.0025:2.329660e-5,.02:2.247112e-5}
def save():
 out['cpu_seconds']=time.process_time()-start;out['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 (BASE/'counted_full.json').write_text(json.dumps(out,indent=2)+'\n')
for sigma in (0.,.0025,.02):
 audit=Audit();t=time.process_time();model=base if sigma==0 else noisy(base,sigma)
 row=dict(sigma=sigma,model_sha256=hashlib.sha256(b''.join(np.asarray(a,dtype=np.float64).tobytes() for a in model)).hexdigest())
 try:
  rays,geometry,_=enumerate_correct(model,audit);row.update(geometry=geometry,correct_simplices=len(rays),geometry_cpu=time.process_time()-t)
  ray_array=np.asarray(rays);signs=np.sign(ray_array.sum(axis=2))
  row['floating_orthant_minimum_signed_coordinate']=float(np.min(ray_array*signs[:,:,None]))
  row['maximum_affine_plane_residual']=float(np.max(abs(np.einsum('ndv,nd->nv',ray_array,signs)-1)))
  value,stats=volume(rays,audit);row.update(estimate=value,integration=stats,cpu_seconds=time.process_time()-t)
  row['cost']=audit.report();row['affine_only_forward_equivalents']=audit.known/65
  row['reference_difference']=abs(value-references[sigma]);row['reference_interval_error_ceiling']=row['reference_difference']+reference_errors[sigma]
  error=stats['conditional_outer_bound']+stats['positive_inner_bound'];lo=max(0,value-error);hi=min(1,value+error)
  variance=min(lo*(1-lo),hi*(1-hi))/row['affine_only_forward_equivalents']
  row['conditional_bound_squared_over_sampling_variance']=error*error/variance
  row['reference_ceiling_squared_over_sampling_variance']=row['reference_interval_error_ceiling']**2/variance
 except Exception as exc:row.update(failure=repr(exc),cost=audit.report(),cpu_seconds=time.process_time()-t)
 out['models'].append(row);save();print(row,flush=True)
 if time.process_time()-start>105:break
out['source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (BASE/'counted.py',BASE/'fixed_tangent.py',BASE/'primitive.py')}
save()
