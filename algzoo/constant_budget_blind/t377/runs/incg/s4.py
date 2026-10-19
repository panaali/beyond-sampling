import numpy as np, sys
from feats import feats
from rfit import relufit, r2
d=np.load("d1.npz"); x=d["x"][:20000].astype(float); h=d["h"][:20000].astype(float)
F=feats(x); names=list(F); M=np.stack([F[n] for n in names],1)
one=np.ones((len(x),1))
for i in range(32):
    y=h[:,i]
    if y.var()<1e-6: print(i,"dead");continue
    sel=[]; out=[]
    for step in range(4):
        best=(-9,None,None)
        for k in range(M.shape[1]):
            if k in sel: continue
            A=np.c_[M[:,sel+[k]],one]; c,r=relufit(A,y,iters=15)
            if r>best[0]: best=(r,k,c)
        sel.append(best[1]); out.append("%.3f"%best[0])
    print(i,out,[names[k] for k in sel],np.round(best[2],2),flush=True)
