import numpy as np
from feats import lib
d=np.load('d1.npz'); x,h=d['x'][:40000],d['h'][:40000]
F=lib(x); names=list(F); A=np.stack([F[n] for n in names],1)
A=(A-A.mean(0))/(A.std(0)+1e-9)
for i in range(32):
    y=h[:,i]-h[:,i].mean(); 
    if y.std()<1e-6: print(i,'const'); continue
    chosen=[]; out=[]
    for step in range(4):
        best=None
        for k in range(A.shape[1]):
            if k in chosen: continue
            X=np.c_[A[:,chosen+[k]],np.ones(len(y))]
            c=np.linalg.lstsq(X,y,rcond=None)[0]; r2=1-((X@c-y)**2).mean()/y.var()
            if best is None or r2>best[0]: best=(r2,k)
        chosen.append(best[1]); out.append(f'{names[best[1]]}:{best[0]:.2f}')
    print(i,' '.join(out))
