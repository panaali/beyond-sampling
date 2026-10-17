"""Independent rank integral for a two-channel identity-memory model."""
from pathlib import Path
import importlib.util,json,math,time
import numpy as np
from scipy.integrate import quad
start=time.process_time();path=Path('/workspace/work/recover-small_h4_4/m16/approaches/middle_target_pair/terminal.py');s=importlib.util.spec_from_file_location('review_middle',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
model=(np.array([1.,-1.]),np.eye(2),np.array([[0.,1.],[-1.,-1.],[1.,0.],[-2.,-2.]]));rows=[]
Phi=lambda x:.5*math.erfc(-x/math.sqrt(2));density=lambda x:math.exp(-x*x/2)/math.sqrt(2*math.pi)
for u in (-1.,0.,1.):
 edges=sorted(set([-6.,0.,6.,u,-u]));raw=np.zeros((2,2))
 # After x then fixed u: a=max(max(x,0)+u,0), b=max(max(-x,0)-u,0).
 # Final winner2 iff y>(b-a)/2, since max(a+y,0)>max(b-y,0).
 for i,(lo,hi) in enumerate([(-6.,u),(u,6.)]):
  for j,(yl,yh) in enumerate([(-6.,u),(u,6.)]):
   def f(x):
    a=max(max(x,0)+u,0);b=max(max(-x,0)-u,0);threshold=(b-a)/2
    return density(x)*max(0.,Phi(yh)-Phi(max(yl,threshold)))
   split=sorted(set([lo,hi]+[v for v in edges if lo<v<hi]))
   # Include threshold crossing y=u and support bounds for each affine branch.
   for a,b in list(zip(split[:-1],split[1:])):
    ta=(max(max(-a,0)-u,0)-max(max(a,0)+u,0))/2;tb=(max(max(-b,0)-u,0)-max(max(b,0)+u,0))/2
    if ta!=tb:
     for y in(yl,yh):
      root=a+(y-ta)*(b-a)/(tb-ta)
      if a<root<b:split.append(root)
   split=sorted(set(split));raw[i,j]=sum(quad(f,a,b,epsabs=1e-13,epsrel=1e-13)[0] for a,b in zip(split[:-1],split[1:]))
 L,U=Phi(u)-Phi(-6),Phi(6)-Phi(u);reference=np.array([raw[0,0]/L**2,(raw[0,1]+raw[1,0])/(2*L*U)]);actual,_=m.MiddleTarget(model,np.zeros(2),u).probabilities();difference=float(np.max(abs(reference-actual)));assert difference<2e-12
 rows.append({'u':u,'independent':reference.tolist(),'author':actual.tolist(),'max_difference':difference})
out={'identity_memory_controls':rows,'cpu_seconds':time.process_time()-start,'scope':'nontrivial default penultimate-label primitive and conditional rank law; no trained model or outer run'}
Path(__file__).with_name('middle_target_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
