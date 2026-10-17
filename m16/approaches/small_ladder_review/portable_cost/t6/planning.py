"""Closed T6 Chebyshev tail and fully charged planning; no mass evaluations."""
import copy,hashlib,json,math,time
from pathlib import Path
import numpy as np
from ..accounting import Audit
from .geometry import determinant
from .jacobi import outer_rule
HERE=Path(__file__).resolve().parent
ORDERS=(4,6,8,10,12,16,20,24,28,32)

def radial_data(ray,audit):
 det=abs(determinant(ray,audit));gram=ray.T@ray
 center=ray.sum(axis=1);center/=math.sqrt(float(center@center));projection=float(np.min(ray.T@center))
 lo=max((1-1e-12)/6,float(gram.min())-1e-12,(projection*projection-1e-12 if projection>0 else 0.))
 hi=max(lo,float(np.diag(gram).max())+1e-12)
 sa=math.sqrt(lo);sb=math.sqrt(hi);s=sa*sb;alpha=(lo+hi)/(2*s);r=(sb-sa)/(sb+sa)
 audit.add('radial_setup',1024)
 return det,s,alpha,r,lo,hi

def tail_bound(data,n,audit):
 det,s,alpha,r,_,_=data;N=n # K=n-1 with matched Jacobi outer weights
 if r==0:return 0.
 one=1-r
 bracket=N*N+3*alpha*N+3*alpha*alpha-1+(2*N+3*alpha)*r/one+r*(1+r)/(one*one)
 tail=(r**N)/(s*s*s*one)*bracket
 value=det/(60*math.pi*math.pi*math.pi)*tail
 # Underflow is immaterial at requested tolerances; keep a conservative floor
 # in the bounded domain n<=32,lo>=1/6 anddet<=1.
 if value==0:value=det*1e-300
 audit.add('closed_tail_trials',120)
 return value

def main():
 start=time.process_time();geo=json.loads((HERE/'geometry_results.json').read_text());cache=HERE/'independent_correct_rays.npz'
 assert geo['complete'] and 'error_type' not in geo
 assert hashlib.sha256(cache.read_bytes()).hexdigest()==geo['ray_cache_sha256']
 rays=np.load(cache)['rays'];norm_error=float(np.max(abs(np.sum(abs(rays),axis=1)-1)))
 mixed=int(np.sum(np.any((rays.min(axis=2)<0)&(rays.max(axis=2)>0),axis=1)))
 assert norm_error<1e-12 and mixed==0,(norm_error,mixed)
 radial_audit=Audit();facts=[radial_data(r,radial_audit) for r in rays]
 out=dict(source_geometry_sha256=geo['ray_cache_sha256'],correct_simplices=len(rays),ray_l1_error=norm_error,mixed_orthant_cones=mixed,radial_cost=radial_audit.report(),plans=[],mass_evaluations=0)
 for target in (1e-5,1e-6,1e-7):
  a=copy.deepcopy(radial_audit);chosen=[];bounds=[];per=target/len(rays);unresolved=0
  for data in facts:
   for n in ORDERS:
    bound=tail_bound(data,n,a)
    if bound<=per:break
   chosen.append(n);bounds.append(bound)
   if bound>per:unresolved+=1
  # Cold setup is charged for every order actually present, not a free cache.
  a.add('rule_order_selection',2*len(ORDERS)*len(rays))
  for n in ORDERS:
   if n in chosen:outer_rule(n,a)
  histogram={str(n):chosen.count(n) for n in ORDERS if n in chosen}
  row=dict(target=target,bound_sum=float(sum(bounds)),unresolved=unresolved,orders=histogram,planned_triangles=sum(n**3 for n in chosen),planning_and_rule_setup_cost=a.report(),full_geometry_plus_planning_ops=geo['production_cost']['known_arithmetic_upper_bound']+a.known)
  out['plans'].append(row)
  np.savez_compressed(HERE/f'plan_{target}.npz',orders=np.array(chosen),bounds=np.array(bounds))
  out['cpu_seconds']=time.process_time()-start;(HERE/'planning_results.json').write_text(json.dumps(out,indent=2)+'\n')
  print(json.dumps({k:row[k] for k in ('target','bound_sum','unresolved','orders','planned_triangles','full_geometry_plus_planning_ops')}),flush=True)
 out['cpu_seconds']=time.process_time()-start;(HERE/'planning_results.json').write_text(json.dumps(out,indent=2)+'\n')
 print('CPU',out['cpu_seconds'],flush=True)
if __name__=='__main__':main()
