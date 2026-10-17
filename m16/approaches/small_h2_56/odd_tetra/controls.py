import json,time,resource
from pathlib import Path
import numpy as np
from scipy.special import roots_jacobi
from primitive import inverse_fifth_tetra,positive_fallback,radial_interval,inverse_fifth_segment
BASE=Path(__file__).parent;start=time.process_time();out={'cases':[]}
def save():
 out['cpu_seconds']=time.process_time()-start;out['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 (BASE/'controls.json').write_text(json.dumps(out,indent=2)+'\n')
def reference(v,n=96):
 x,w=roots_jacobi(n,2,0);x=(x+1)/2;w=w/8;z,t=roots_jacobi(n,1,0);z=(z+1)/2;t=t/4
 a,b=np.meshgrid(x,z,indexing='ij');weight=(w[:,None]*t[None,:]).ravel();y=np.stack([a,(1-a)*b],axis=-1).reshape(-1,2);r=((1-a)*(1-b)).ravel();prefix=y@v[:2]
 return float(weight@inverse_fifth_segment(prefix+r[:,None]*v[2],prefix+r[:,None]*v[3]))
e=np.eye(5);cases=[('identity',e[:4],2*np.pi/3),('equal',np.tile(e[0],(4,1)),1/6),('collinear',np.arange(1,5)[:,None]*e[0],None)]
for k in range(12):
 v=np.array([[1+((i+1)*(j+2)+k*(i+3))%11 for j in range(5)] for i in range(4)],float);v/=v.sum(axis=1)[:,None]
 if k%3==0:v=v[0]+10.**(-k-1)*(v-v[0])
 v*=np.array([1,-1,1,-1,1]);cases.append(('deterministic_'+str(k),v,None))
for name,v,truth in cases:
 ref=reference(v);normal=np.sign(v.sum(axis=0)) if np.max(abs(np.sum(abs(v),axis=1)-1))<1e-12 else None
 value,stats=inverse_fifth_tetra(v,known_plane_normal=normal);stats.pop('quadrature_error_bounds');row=dict(name=name,estimate=float(value),reference=ref,relative_error=float(abs(value-ref)/abs(ref)),analytic_truth=truth,stats=stats);out['cases'].append(row);save()
 assert row['relative_error']<2e-9
 if truth is not None:assert abs(value-truth)<2e-10
 for order in ((3,2,1,0),(1,3,0,2)):
  changed,_=inverse_fifth_tetra(v[list(order)],known_plane_normal=normal);assert abs(changed-value)/abs(value)<2e-9
 for scale in (1e-3,1e3):
  changed,_=inverse_fifth_tetra(v*scale,known_plane_normal=None,absolute_tolerance=1e-11/scale**5);assert abs(changed*scale**5-value)/abs(value)<2e-9
print('controls',len(out['cases']),'max relative',max(c['relative_error'] for c in out['cases']),'CPU',time.process_time()-start,flush=True);save()
