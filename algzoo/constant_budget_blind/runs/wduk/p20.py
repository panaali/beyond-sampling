import numpy as np, probe, itertools
from fitlib import *
x=np.load('x1.npy')[:30000];h=np.load('h1.npy')[:30000]
xt=np.random.default_rng(7).standard_normal((30000,10));ht=probe.hidden(xt)
def feats(x,k):
    w=-np.sort(-x[:,:k],1);F=pf(x)
    return {'m_w':w[:,0],'s_w':w[:,1],'t_w':w[:,2],'r2':np.maximum(x[:,2],0),'M':F['M'],'S':F['S'],'T':F['T'],'sumrelu':np.maximum(x[:,:k],0).sum(1),'x2':x[:,2]}
for i in (2,13,9):
  out=[]
  for k in range(3,9):
    D=feats(x,k);Dt=feats(xt,k)
    for cols in itertools.combinations(D.keys(),3):
        X=np.c_[[D[c] for c in cols]].T;Xt=np.c_[[Dt[c] for c in cols]].T
        c,*_=np.linalg.lstsq(X,h[:,i],rcond=None)
        r=least_squares(lambda th:np.maximum(0,X@th)-h[:,i],c)
        out.append((r2(np.maximum(0,Xt@r.x),ht[:,i]),k,cols,np.round(r.x,1)))
  out.sort(key=lambda t:-t[0])
  for o in out[:4]: print(i,'%.3f'%o[0],o[1:])
