import numpy as np
from feats import feats
d=np.load('d1.npz');x=d['x'][:40000];H=d['H'][:40000]
F=feats(x);names=list(F);A=np.stack([F[n] for n in names],1)
for i in range(16):
    y=H[:,i];sel=[];res=[]
    for k in range(5):
        best=None
        for j in range(A.shape[1]):
            if j in sel: continue
            B=A[:,sel+[j]];c,*_=np.linalg.lstsq(B,y,rcond=None);r=1-((y-B@c)**2).mean()/y.var()
            if best is None or r>best[0]: best=(r,j,c)
        sel.append(best[1]);res.append(f"{names[best[1]]}({best[0]:.3f})")
    print(i," ".join(res)," coef",np.round(best[2],2))
