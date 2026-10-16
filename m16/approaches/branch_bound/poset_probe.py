# diagnostic only: certified fraction of boxes of half-width h by poset type (top-k chain + star), absolute gap coords
import sys; sys.path.insert(0,'/home/user/clean-neurons/m16/approaches/branch_bound')
import numpy as np, core
from core import T, D
def chainM(order, k):
    M=np.zeros((T,D)); M[:,0]=1
    for i in range(k-1):            # x_{order[i]} = v + sum_{l>=i} d_l, l<k-1
        for l in range(i,k-1): M[order[i],1+l]=1
    for idx,j in enumerate(order[k:]): M[j,k+idx]=-1
    return M
rng=np.random.default_rng(1); N=3000
x=rng.standard_normal((N,T)); od=np.argsort(-x,1)
for k in (2,3,5,10):
    M=np.array([chainM(o,k) for o in od]); w=np.array([np.linalg.lstsq(m,xx,rcond=None)[0] for m,xx in zip(M,x)])
    core.MS=M; core.TGT=od[:,1]
    for h in (0.1,0.03,0.01):
        hw=np.full((N,D),h); wl=w-hw; wu=w+hw; wl[:,1:]=np.maximum(wl[:,1:],0)
        lb,ub,_=core.bounds(np.arange(N),wl,wu,inter='crown')
        fin=np.isfinite(ub); c=(lb>0).all(1); wr=((ub<0)&fin).any(1)
        print(k,h,'correct',c.mean().round(3),'wrong',wr.mean().round(3))
