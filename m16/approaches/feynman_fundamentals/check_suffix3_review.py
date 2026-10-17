import sys,time,json,math
from pathlib import Path
import numpy as np
ROOT=Path('/workspace/work/recover-small_h2_56/m16/approaches/h2_boundary_mechanism')
sys.path.insert(0,str(ROOT))
from inventory import load
OUT=Path(__file__).parent
base,_=load(4);rng=np.random.default_rng(47021)
wi,wh,wo=tuple(w+.02*np.sqrt(np.mean(w*w))*rng.normal(size=w.shape) for w in base)
start=time.process_time();rows=[]
for n in (128,256):
 z,weights=np.polynomial.legendre.leggauss(n);phi=(np.arange(2*n)+.5)*2*math.pi/(2*n)
 points=np.stack(np.broadcast_arrays(np.sqrt(1-z[:,None]**2)*np.cos(phi),np.sqrt(1-z[:,None]**2)*np.sin(phi),z[:,None]),axis=-1).reshape(-1,3)
 angular=np.broadcast_to(weights[:,None]/(4*n),(n,2*n)).ravel();value=control=0.
 for k in range(0,len(points),65536):
  x=points[k:k+65536];h=np.zeros((len(x),2))
  for step in range(3):h=np.maximum(h@wh.T+x[:,step,None]*wi,0)
  label=np.argmax(h@wo.T,axis=1);rank=np.argsort(x,axis=1);upper=x[np.arange(len(x)),rank[:,-1]];second=x[np.arange(len(x)),rank[:,-2]]
  def f(a):return .5+(np.arctan(a)+a/(1+a*a))/math.pi
  fu,fs=f(upper),f(second);truth=fu-fs
  truth=np.where(label==1+rank[:,-1],1-fu,truth)
  truth=np.where(label==1+rank[:,-2],fs,truth)
  truth=np.where(label==1+rank[:,0],0,truth)
  value+=float(angular[k:k+len(x)]@truth);control+=float(angular[k:k+len(x)]@(fu-fs))
 rows.append(dict(n=n,nodes=len(points),surrogate=value,constant_control=control,constant_error=control-.25))
 print(rows[-1],flush=True)
result=dict(rows=rows,cpu_seconds=time.process_time()-start,model_input_sampling=0,three_step_quadrature_forwards=sum(r['nodes'] for r in rows),full_reference=.2636679118915033)
(OUT/'suffix3_review_results.json').write_text(json.dumps(result,indent=2)+'\n')
