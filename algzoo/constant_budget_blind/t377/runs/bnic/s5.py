import numpy as np, feats
x=np.load('x1.npy')[:40000]; h=np.load('h1.npy')[:40000]
F=feats.pool(x); names=list(F); A=np.stack([F[k] for k in names],1)
A=(A-A.mean(0))/(A.std(0)+1e-9)
for i in range(16):
    y=h[:,i]-h[:,i].mean(); sel=[]; out=[]
    for step in range(5):
        best=None
        for k in range(A.shape[1]):
            if k in sel: continue
            B=A[:,sel+[k]]; c,*_=np.linalg.lstsq(B,y,rcond=None); r=1-((y-B@c)**2).mean()/y.var()
            if best is None or r>best[0]: best=(r,k)
        sel.append(best[1]); out.append(f'{names[best[1]]}:{best[0]:.3f}')
    print(i,' '.join(out))
