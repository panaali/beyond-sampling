import numpy as np, sys
from feats2 import feats, P5
from rfit import relufit, r2
d=np.load("d1.npz"); x=d["x"][:12000].astype(float); h=d["h"][:12000].astype(float)
F=feats(x); names=list(F); M=np.stack([F[n] for n in names],1)
for i in range(32):
    y=h[:,i]
    if y.var()<1e-3: print(i,"dead");continue
    Mi=M; nm=list(names)
    if P5.get(i):
        P=P5[i]; Q=[j for j in range(9) if j not in P]
        extra=[x[:,P].max(1)]; nm2=["maxP"]
        if Q: extra.append(x[:,Q].max(1)); nm2.append("maxQ")
        Mi=np.c_[M,np.stack(extra,1)]; nm=nm+nm2
    sel=[]; out=[]
    for step in range(4):
        best=(-9,None,None)
        for k in range(Mi.shape[1]):
            if k in sel: continue
            c,r=relufit(Mi[:,sel+[k]],y,iters=12)
            if r>best[0]: best=(r,k,c)
        sel.append(best[1]); out.append("%.3f"%best[0])
    print(i,out,[nm[k] for k in sel],np.round(best[2],2),flush=True)
