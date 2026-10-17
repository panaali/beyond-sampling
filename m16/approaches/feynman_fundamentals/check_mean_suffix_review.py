"""Independent direct product Gaussian quadrature; no affine-region engine."""
import sys,json,time
from pathlib import Path
import numpy as np
from scipy.special import roots_hermitenorm,ndtr,ndtri
sys.path.insert(0,'/workspace/work/recover-small_h2_56/m16/approaches/small_h2_56')
from models import load
start=time.process_time()
base,sha=load(4);rng=np.random.default_rng(47021)
wi,wh,wo=tuple(w+.02*np.sqrt(np.mean(w*w))*rng.normal(size=w.shape) for w in base)
# Derive the exact second-state mean directly from the two first-input signs.
mu=sum(np.hypot(wh@np.maximum(sign*wi,0),wi)+wh@np.maximum(sign*wi,0) for sign in (1,-1))/(2*np.sqrt(2*np.pi))
rows=[]
for n in (128,256,1024,2048):
    if n<1000:
        yn,yw=roots_hermitenorm(n);xn,xw=roots_hermitenorm(n+1)
        yw=yw/np.sqrt(2*np.pi);xw=xw/np.sqrt(2*np.pi)
    else:
        yn=ndtri((np.arange(n)+.5)/n);xn=ndtri((np.arange(n+1)+.5)/(n+1))
        yw=np.full(n,1/n);xw=np.full(n+1,1/(n+1))
    for mode,initial in [('zero',np.zeros(2)),('exact2',mu)]:
        total=constant=0.
        chunk=65536//len(xn)
        for begin in range(0,n,chunk):
            y=np.repeat(yn[begin:begin+chunk],len(xn));x=np.tile(xn,len(yn[begin:begin+chunk]))
            weights=(yw[begin:begin+chunk,None]*xw[None,:]).ravel()
            h=np.maximum(initial@wh.T+y[:,None]*wi,0)
            h=np.maximum(h@wh.T+x[:,None]*wi,0)
            label=np.argmax(h@wo.T,axis=1)
            high=ndtr(np.maximum(x,y));low=ndtr(np.minimum(x,y))
            older=.5-high+high*high-.5*low*low
            larger=2*high*(1-high);smaller=low*low
            first=np.where(y>x,larger,smaller);last=np.where(x>y,larger,smaller)
            p=np.where(label<2,older,np.where(label==2,first,last))
            total+=float(weights@p);constant+=float(weights@older)
        rows.append(dict(rule="Gaussian" if n<1000 else "CDF midpoint",order_y=n,order_x=n+1,mode=mode,estimate=total,constant_label0=constant,constant_error=constant-.25))
out=dict(checkpoint_sha256=sha,weights=[wi.tolist(),wh.tolist(),wo.tolist()],exact_prefix_mean=mu.tolist(),rows=rows,cpu_seconds=time.process_time()-start,direct_two_step_forwards=2*sum(n*(n+1) for n in (128,256,1024,2048)),method='deterministic Gaussian and uniform-CDF product quadrature; no convergence certificate')
Path(__file__).with_name('mean_suffix_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
