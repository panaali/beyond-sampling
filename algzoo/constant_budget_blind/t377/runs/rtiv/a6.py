import numpy as np
from feats import feats
d=np.load('d1.npz');x=d['x'][:40000];H=d['H'][:40000]
F=feats(x);F['M']=x.max(1);F['sum']=x.sum(1);F['S']=F.pop('s1')
names=list(F);A=np.stack([F[n] for n in names],1)
np.set_printoptions(precision=1,suppress=True,linewidth=220)
for i in range(16):
    y=H[:,i]
    B=np.c_[F['M'],x];c,*_=np.linalg.lstsq(B,y,rcond=None);rr=1-((y-B@c)**2).mean()/y.var()
    sel=[names.index('M')];res=[]
    for k in range(4):
        best=None
        for j in range(A.shape[1]):
            if j in sel: continue
            B=A[:,sel+[j]];cc,*_=np.linalg.lstsq(B,y,rcond=None);r=1-((y-B@cc)**2).mean()/y.var()
            if best is None or r>best[0]: best=(r,j,cc)
        sel.append(best[1]);res.append(f"{names[best[1]]}({best[0]:.3f})")
    print(i,f"Mx:{rr:.3f}",c,"| M+",' '.join(res),np.round(best[2],1))
