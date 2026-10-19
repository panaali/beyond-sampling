import numpy as np, probe
from lib import base
from rfit import relufit
rng=np.random.default_rng(5)
x=rng.standard_normal((200000,10)); x=x[(x.max(1)<0)]  # all negative, natural conditional
x=np.r_[x, -np.abs(rng.standard_normal((20000,10)))][:20000]
h=np.asarray(probe.hidden(x.astype(np.float32))).astype(float)
F=base(x); nm=list(F); M=np.stack([F[n] for n in nm],1)
for i in [16,25]:
    y=h[:,i]; print(i,"act",(y>0).mean(),"mean",y.mean())
    sel=[];rs=[]
    for s in range(4):
        best=(-9,None,None)
        for k in range(M.shape[1]):
            if k in sel: continue
            c,r=relufit(M[:,sel+[k]],y,iters=10)
            if r>best[0]: best=(r,k,c)
        sel.append(best[1]); rs.append(round(best[0],3))
    print(rs,[nm[k] for k in sel],np.round(best[2],2))
