"""Retain exact-real h2 sector moments as separate components; floating implementation."""
import hashlib,json,math,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;P=json.loads((HERE/'protocol.json').read_text())
for path,sha in P['source_pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
SRC=next(Path(p).parent for p in P['source_pins'] if p.endswith('/prefix_covariance.py'))
sys.path.insert(0,str(SRC))
from prefix_covariance import prefix2
from cumprop_snapshot import rnn_step_inject

def components(wi,wh):
    _,_,metadata=prefix2(wi,wh);weights=[];means=[];covs=[];rows=[]
    aa={s:wh@np.maximum(s*wi,0) for s in (1,-1)}
    for sector in metadata['sectors']:
        lo,hi=sector['lo'],sector['hi'];width=hi-lo;mid=(hi+lo)/2;d=width/2;c,s=math.cos(mid),math.sin(mid)
        B=np.column_stack((aa[sector['sign']],wi));active=B@np.array([c,s])>0;A=B*active[:,None]
        # Conditional first angular moment; sinc form avoids dividing two tiny quantities.
        angular_mean=(math.sin(d)/d)*np.array([c,s])
        # Conditional angular sin(v)^2, using the same stable series as the aggregate kernel.
        if d<.001:perp=d*d*(1/3+d*d*(-1/15+d*d*(2/315+d*d*(-1/2835))))
        else:perp=.5-math.sin(2*d)/(4*d)
        R=np.array([[c,-s],[s,c]]);J=(R*np.array([1-perp,perp]))@R.T
        mu=math.sqrt(math.pi/2)*(A@angular_mean);raw=2*(A@J@A.T);cov=raw-np.outer(mu,mu)
        weights.append(width/(2*math.pi));means.append(mu);covs.append(cov)
        rows.append(dict(**sector,mass=weights[-1],min_covariance_eigenvalue=float(np.linalg.eigvalsh(cov)[0]),max_covariance_entry=float(np.max(abs(cov)))))
    return np.array(weights),np.array(means),np.array(covs),rows

def combine(weights,means,covs):
    mu=np.einsum('b,bi->i',weights,means);delta=means-mu
    cov=np.einsum('b,bij->ij',weights,covs+delta[:,:,None]*delta[:,None,:])
    return mu,cov

def continuation(wi,wh):
    weights,mu,cov,meta=components(wi,wh);Q=np.outer(wi,wi);trace=[combine(weights,mu,cov)];states=[dict(mean=mu.tolist(),covariance=cov.tolist())]
    for t in (3,4):
        out=[rnn_step_inject(m,c,wh,Q,'gauss') for m,c in zip(mu,cov)]
        mu=np.array([v[0] for v in out]);cov=np.array([v[1] for v in out]);trace.append(combine(weights,mu,cov));states.append(dict(mean=mu.tolist(),covariance=cov.tolist()))
    return trace,dict(weights=weights.tolist(),components=meta,states=states,closure_calls=2*len(weights))
