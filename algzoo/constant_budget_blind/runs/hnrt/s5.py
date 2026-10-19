import numpy as np, probe, itertools
from util import *
rng=np.random.default_rng(3)
x=rng.standard_normal((100000,10)); H=probe.hidden(x)
o,s=feats(x); N=len(x)
oh1=np.zeros((N,10)); oh1[np.arange(N),o[:,0]]=s[:,0]
Fflex=np.hstack([oh1,s[:,1:2],s[:,2:3],x[:,9:10]])
def km(v,k):
    # exhaustive best 1D partition into k contiguous groups of sorted values
    idx=np.argsort(v); best=None
    for cuts in itertools.combinations(range(1,10),k-1):
        gs=np.split(idx,cuts); sse=sum(((v[g]-v[g].mean())**2).sum() for g in gs)
        if best is None or sse<best[0]: best=(sse,gs)
    return best[1]
for i in range(16):
    h=H[:,i]; w=relufit(Fflex,h); rf=r2(np.maximum(Fflex@w,0),h)
    gs=km(w[:10],3)
    F=np.stack([oh1[:,g].sum(1) for g in gs]+[s[:,1],x[:,9]],1)
    w2=relufit(F,h); rg=r2(np.maximum(F@w2,0),h)
    print(i,"flex",round(rf,3),"grp3",round(rg,3),[sorted(g.tolist()) for g in gs],np.round(w2,1))
