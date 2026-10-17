"""Independent Fraction line minima for fixed crossings and degeneracies."""
from fractions import Fraction as F
import itertools,json
from pathlib import Path
import numpy as np
from active import line,fit
HERE=Path(__file__).parent

def exact(a,b,y):
 edges=sorted(set([F(0),F(1)]+[-x/z for x,z in zip(a,b) if z and 0<-x/z<1]));candidates=list(edges)
 for left,right in zip(edges[:-1],edges[1:]):
  middle=(left+right)/2;active=[i for i in range(len(a)) if a[i]+b[i]*middle>0];A=sum((b[i]**2 for i in active),F(0));B=sum((b[i]*(a[i]-y[i]) for i in active),F(0))
  if A:candidates.append(min(right,max(left,-B/A)))
 def loss(t):return sum(((max(F(0),x+t*z)-v)**2 for x,z,v in zip(a,b,y)),F(0))
 return min(loss(t) for t in candidates)

cases=[]
for shift in range(75):
 a=[F(((i*3+shift)%7)-3,2) for i in range(7)];b=[F(((i*5+shift*3)%9)-4,2) for i in range(7)];y=[F((i+shift)%4,2) for i in range(7)];cases.append((a,b,y))
cases.extend([([F(0)]*3,[F(0)]*3,[F(1)]*3),([F(-1),F(-2),F(1)],[F(2),F(4),F(-2)],[F(1)]*3),([F(-1)]*3,[F(0)]*3,[F(0)]*3),([F(0)]*3,[F(-1),F(0),F(1)],[F(1)]*3)])
errors=[]
for a,b,y in cases:
 phi=np.array(list(zip(a,b)),dtype=float);target=np.array(y,dtype=float);_,meta=line(phi,target,np.array([1.,0.]),np.array([0.,1.]));reference=float(exact(a,b,y));errors.append(abs(meta['loss']-reference));assert meta['loss']<=meta['old_loss'] and errors[-1]<1e-11
phi=np.array([[1.,i/10] for i in range(-10,11)]);truth=np.array([[1.,.5],[.2,-.4]]);h=np.maximum(phi@truth,0);beta,meta=fit(phi,h,np.ones((2,2)),1e-10)
assert meta['converged'] and meta['train_mse']<1e-20
out=dict(rational_cases=len(cases),max_line_minimum_error=max(errors),synthetic_fit_mse=meta['train_mse'],synthetic_fit_gradient=meta['original_gradient_inf'],all_nonincreasing=True)
(HERE/'active_controls.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
