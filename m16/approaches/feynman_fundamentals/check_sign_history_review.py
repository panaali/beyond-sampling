import time,json,sys
from pathlib import Path
import numpy as np
from scipy.special import ndtri
sys.path.insert(0,'/workspace/work/recover-small_h2_56/m16/approaches/small_h2_56')
from models import load
start=time.process_time();base,sha=load(4);rng=np.random.default_rng(47021)
wi,wh,wo=[v+.02*np.sqrt(np.mean(v*v))*rng.standard_normal(v.shape) for v in base]
means={sign:(np.hypot(wh@np.maximum(sign*wi,0),wi)+wh@np.maximum(sign*wi,0))/np.sqrt(2*np.pi) for sign in (-1,1)}
rows=[]
for n in (1024,2048):
 qy=(np.arange(n)+.5)/n;qx=(np.arange(n+1)+.5)/(n+1);yn=ndtri(qy);xn=ndtri(qx);count=n*(n+1)
 correct=shuffled=constant=0.;minimum=1.;defect=0.
 for sign in (-1,1):
  for begin in range(0,n,65536//(n+1)):
   end=min(n,begin+65536//(n+1));y=np.repeat(yn[begin:end],n+1);x=np.tile(xn,end-begin)
   fy=np.repeat(qy[begin:end],n+1);fx=np.tile(qx,end-begin);lo=np.minimum(fx,fy);hi=np.maximum(fx,fy)
   Glo=np.maximum(0,2*lo-1) if sign>0 else np.minimum(1,2*lo)
   Ghi=np.maximum(0,2*hi-1) if sign>0 else np.minimum(1,2*hi)
   smaller=Glo*lo;larger=hi+Ghi-2*Ghi*hi
   a=np.maximum(lo,.5) if sign>0 else np.minimum(lo,.5)
   b=np.maximum(hi,.5) if sign>0 else np.minimum(hi,.5)
   special=(b-a)*(b+a)
   a=np.maximum(hi,.5) if sign>0 else np.minimum(hi,.5);b=1. if sign>0 else .5
   special+=(b-a)*(2-b-a)
   regular=1-smaller-larger-special
   first=np.where(y>x,larger,smaller);last=np.where(x>y,larger,smaller)
   probs=np.column_stack((special,regular,first,last));minimum=min(minimum,float(probs.min()));defect=max(defect,float(abs(probs.sum(axis=1)-1).max()))
   constant+=.5*float(special.sum())/count
   for swap in (False,True):
    mu=means[-sign if swap else sign]
    h=np.maximum(mu@wh.T+y[:,None]*wi,0);h=np.maximum(h@wh.T+x[:,None]*wi,0)
    label=(h@wo.T).argmax(axis=1);value=.5*float(probs[np.arange(len(label)),label].sum())/count
    if swap:shuffled+=value
    else:correct+=value
 rows.append(dict(n=n,correct_sign=correct,shuffled_means=shuffled,minimum_probability=minimum,partition_defect=defect,constant_label0=constant,forward_calls=4*count))
out=dict(checkpoint_sha256=sha,conditional_means={str(k):v.tolist() for k,v in means.items()},rows=rows,cpu_seconds=time.process_time()-start,scope='deterministic CDF product quadrature, independent rank polynomials/direct forward; no error enclosure')
Path(__file__).with_name('sign_history_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
