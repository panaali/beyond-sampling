import numpy as np, feats, sys
x=np.load('x1.npy')[:30000]; h=np.load('h1.npy')[:30000]
F=feats.pool2(x); names=list(F); A=np.stack([F[k] for k in names],1)
for i in range(16):
    y=h[:,i]; sel=[]; out=[]
    for step in range(5):
        best=None
        for k in range(A.shape[1]):
            if k in sel: continue
            B=A[:,sel+[k]]; c,*_=np.linalg.lstsq(B,y,rcond=None); r=1-((y-B@c)**2).mean()/y.var()
            if best is None or r>best[0]: best=(r,k,c)
        sel.append(best[1]); out.append(f'{names[best[1]]}:{best[0]:.3f}')
    print(i,' '.join(out),best[2].round(1))
