"""Local partition adapter; input ties are assigned by owner(), not closed cones."""
import numpy as np
import estT_snapshot as engine
from m43 import Ops

def owner(x):
    p=sorted(range(len(x)),key=lambda i:(-x[i],i))
    return p[0],p[1],1 if x[0]>=0 else -1

def quadratic(T,ops):
    E=np.eye(T);out=[]
    for sigma in (1.,-1.):
      for a in range(T):
       for b in range(T):
        if a==b:continue
        normals=[E[a]-E[b]]+[E[b]-E[k] for k in range(T) if k not in (a,b)]+[sigma*E[0]]
        c=engine.Cone.from_normals(normals)
        ops.add(T**3+T*T*(T+3)+T*T) # labeled construction proxy, not audited complete circuit
        out.append(dict(sigma=sigma,truth=b,n_true=E[a]-E[b],cone=c,pair=(a,b)))
    return out

def nearest(T,ops):
    E=np.eye(T);out=[]
    for h in quadratic(T,ops):
        a,b=h['pair'];ks=[k for k in range(T) if k not in(a,b)]
        c=h['cone']
        for k in ks:
            if c is not None:c,_=engine.clip(c,2*E[b]-E[a]-E[k],ops)
        if c is not None:out.append(dict(h,cone=c))
        for k in ks:
            c=h['cone']
            for j in ks:
                if j!=k and c is not None:c,_=engine.clip(c,E[k]-E[j],ops)
            if c is not None:c,_=engine.clip(c,E[a]+E[k]-2*E[b],ops)
            if c is not None:out.append(dict(h,cone=c,n_true=E[b]-E[k]))
    return out

def run(model,kind,mode):
    ops=Ops();T=model[2].shape[0]
    if kind=='old':
        hs=engine.halves(T); setup=None # historical setup was uncounted
    else:
        hs=(quadratic if kind=='quadratic' else nearest)(T,ops);setup=ops.n
    engine._HALVES[T]=hs
    engine.STATS.update({k:0 for k in engine.STATS})
    val=engine.estT(model,ops=ops,mode=mode)
    return dict(value=val,ops_proxy=ops.n,setup_proxy=setup,families=len(hs),stats=engine.STATS.copy())
