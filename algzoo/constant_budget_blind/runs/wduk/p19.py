import numpy as np, probe, itertools
from fitlib import *
x=np.load('x1.npy')[:30000];h=np.load('h1.npy')[:30000]
xt=np.random.default_rng(7).standard_normal((30000,10));ht=probe.hidden(xt)
def feats(x,A):
    F=pf(x);inA=np.isin(F['P'],A)&np.isin(F['Q'],A)
    return np.c_[F['S']*inA, np.maximum(x[:,2],0), F['M'], F['S'], F['T']]
res=[]
for k in range(3,8):
  for A in itertools.combinations(range(9),k):
    if 2 not in A: continue
    X=feats(x,list(A));hi=h[:,2]
    c,*_=np.linalg.lstsq(X,hi,rcond=None)
    res.append((((X@c-hi)**2).mean(),A))
res.sort()
for e,A in res[:4]:
    X=feats(x,list(A));Xt=feats(xt,list(A))
    for cols in ([0,1,2,3,4],[0,1,3],[0,1,2,3],[0,1,4]):
        c,*_=np.linalg.lstsq(X[:,cols],h[:,2],rcond=None)
        r=least_squares(lambda th:np.maximum(0,X[:,cols]@th)-h[:,2],c)
        print(A,cols,'%.3f'%r2(np.maximum(0,Xt[:,cols]@r.x),ht[:,2]),np.round(r.x,1))
