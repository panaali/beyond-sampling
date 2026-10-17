import json, time, resource
from pathlib import Path
import numpy as np
from counted import Audit, PositiveFallback, gauss_rule, primitive, tail
from primitive import inverse_fifth_tetra, positive_tail_bound
BASE=Path(__file__).parent
start=time.process_time();audit=Audit();fallback=PositiveFallback(audit)
x,w=gauss_rule(24,audit);out={'cases':[],'tail_checks':[]}
cases=[('identity',np.eye(5)[None])]
for name in ('t5_cones','t5_sigma0.0025_cones','t5_sigma0.02_cones'):
 a=np.load(BASE.parent/(name+'.npz'))['rays'];ids=np.linspace(0,len(a)-1,32,dtype=int)
 cases.append((name,np.swapaxes(a[ids],1,2)))
p=np.array([.8,.05,.05,.05,.05])
for k in range(2,9):cases.append(('asymmetric_'+str(k),np.vstack([p,p+10.**-k*(np.eye(5)[:4]-p)])[None]))
cases.append(('equal',np.tile(p,(1,5,1))))
for name,v in cases:
 for reflected in (False,True):
  if reflected:v=v[:,[0,4,2,1,3]]*np.array([-1,1,-1,1,-1])
  endpoints=x[None,:,None,None]*v[:,0,None,None]+(1-x[None,:,None,None])*v[:,None,1:]
  actual,stats=primitive(v,x,1e-11,audit,fallback)
  reference,_=inverse_fifth_tetra(endpoints,known_plane_normal=np.sign(v.sum(1))[:,None])
  relative=float(np.max(abs(actual-reference)/abs(reference)))
  stats.pop('quadrature_error_bounds')
  out['cases'].append(dict(name=name,reflected=reflected,maximum_relative_difference=relative,stats=stats))
  assert relative<2e-9,out['cases'][-1]
for a,b in ((.2,1),(.4,.8),(.65,.65)):
 for n in (2,4,8,16,32):
  actual=tail(a,b,n,audit);reference=positive_tail_bound(a,b,2.5,n)
  error=abs(actual-reference)/max(reference,1e-300)
  out['tail_checks'].append(dict(lower=a,upper=b,degree=n,relative_difference=error))
  assert error<1e-13
out['cost']=audit.report();out['cpu_seconds']=time.process_time()-start
out['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
(BASE/'counted_controls.json').write_text(json.dumps(out,indent=2)+'\n')
print('controls',len(out['cases']),'maxrelative',max(c['maximum_relative_difference'] for c in out['cases']),'CPU',out['cpu_seconds'])
