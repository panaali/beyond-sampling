import numpy as np, itertools
from util import relufit
def km(v,k):
    idx=np.argsort(v); best=None
    for cuts in itertools.combinations(range(1,10),k-1):
        gs=np.split(idx,cuts); sse=sum(((v[g]-v[g].mean())**2).sum() for g in gs)
        if best is None or sse<best[0]: best=(sse,gs)
    return [sorted(g.tolist()) for g in best[1]]
def base(x):
    N=len(x); o=np.argsort(-x,1); s=np.take_along_axis(x,o,1)
    D={}
    for j in range(10): D['x%d'%j]=x[:,j]
    for k in range(3): D['s%d'%(k+1)]=s[:,k]
    for t in range(1,10):
        ps=-np.sort(-x[:,:t+1],1)
        D['P%d'%t]=ps[:,0]; D['R%d'%t]=ps[:,1]
        if t>=2: D['T%d'%t]=ps[:,2]
    for t in range(1,9):
        D['Q%d'%t]=x[:,t:].max(1)
    for k in range(2):
        for j in range(10): D['s%d@%d'%(k+1,j)]=s[:,k]*(o[:,k]==j)
        for c in range(3): D['s%d@m%d'%(k+1,c)]=s[:,k]*(o[:,k]%3==c)
    for c in range(3): D['M%d'%c]=x[:,c::3].max(1)
    for j in range(1,10): D['new%d'%j]=np.maximum(x[:,j]-x[:,:j].max(1),0)
    return D,o,s
def groups_for(h,x,o,s):
    N=len(x); out={}
    for k in range(2):
        oh=np.zeros((N,10)); oh[np.arange(N),o[:,k]]=s[:,k]
        F=np.hstack([oh,s[:,2:3],x[:,9:10]]); w=relufit(F,h,10)[:10]
        for kk in (2,3):
            for g in km(w,kk):
                out['s%d@{%s}'%(k+1,','.join(map(str,g)))]=oh[:,g].sum(1)
    return out
