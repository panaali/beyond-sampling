"""Independent scalar radial quadrature and full sphere/constant controls."""
import json,math,time
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from suffix3 import Audit,radial,calculate
start=time.process_time();out={'radial':[],'constants':[]}
for n in (1,2):
 for a in (-1.,-.5,0.,.5,1.):
  closed=radial(np.array([a]),n,Audit())[0][0]
  ref=quad(lambda r:math.sqrt(2/math.pi)*r*r*math.exp(-r*r/2)*ndtr(a*r)**n,0,12,epsabs=1e-13,epsrel=1e-13)[0]
  out['radial'].append(dict(n=n,a=a,closed=closed,reference=ref,error=abs(closed-ref)))
for t in (4,5):
 for label in range(t):
  wo=np.zeros((t,2));wo[label]=1
  model=(np.array([1.,-1.]),np.zeros((2,2)),wo)
  p,s=calculate(model,8,True);q,r=calculate(model,16,True)
  out['constants'].append(dict(t=t,label=label,value=p,reference=1/t,error=abs(p-1/t),refinement=abs(p-q),stats=s))
out['cpu_seconds']=time.process_time()-start
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
print('radial',max(r['error'] for r in out['radial']),'constant',max(r['error'] for r in out['constants']),
 'refinement',max(r['refinement'] for r in out['constants']),'mass',max(abs(r['stats']['angular_mass']-1) for r in out['constants']),
 'mismatches',sum(r['stats']['midpoint_label_mismatches'] for r in out['constants']),'cpu',out['cpu_seconds'],flush=True)
