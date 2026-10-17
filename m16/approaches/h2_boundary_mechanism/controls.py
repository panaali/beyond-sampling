import json,math,time
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from suffix import radial,calculate,Audit
BASE=Path(__file__).parent;start=time.process_time();out={'radial':[],'constant':[]}
for n in (2,3):
 for a in (-1.,-.5,0.,.5,1.):
  got=float(radial(np.array([a]),n,Audit())[0][0])
  reference,error=quad(lambda r:r*math.exp(-r*r/2)*ndtr(a*r)**n,0,np.inf,epsabs=1e-13,epsrel=1e-13)
  out['radial'].append(dict(a=a,n=n,value=got,reference=reference,difference=abs(got-reference)))
  assert abs(got-reference)<2e-13
for t in (4,5):
 for label in range(t):
  wi=np.array([1.,-1.]);wh=np.array([[.2,.7],[-.4,.1]]);wo=np.zeros((t,2));wo[label]=1
  # All positive hidden states have this label, but zero states still choose0.
  # Zero recurrence excludes positive-mass dead states for mixed input signs.
  wh[:]=0
  value,stats=calculate((wi,wh,wo))
  out['constant'].append(dict(t=t,label=label,value=value,error=abs(value-1/t)))
  assert abs(value-1/t)<2e-9
out['cpu_seconds']=time.process_time()-start
(BASE/'controls.json').write_text(json.dumps(out,indent=2)+'\n')
print('max radial',max(r['difference'] for r in out['radial']),'max constant',max(r['error'] for r in out['constant']),'CPU',out['cpu_seconds'])
