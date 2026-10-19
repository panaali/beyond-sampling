import numpy as np
d=np.load('d1.npz');x=d['x'][:40000];H=d['H'][:40000]
o=np.argsort(-x,1);am=o[:,0];M=x.max(1);S=-np.sort(-x,1)[:,1]
ev=(am%2==0).astype(float)
pm=np.maximum.accumulate(x,1)
s8=-np.sort(-x[:,:9],1); s7=-np.sort(-x[:,:8],1)
F={'M':M,'S':S,'evM':ev*M,'evS':ev*S,'pm6':pm[:,6],'pm7':pm[:,7],'pm8':pm[:,8],'S8':s8[:,1],'S7':s7[:,1],'sum':x.sum(1)}
for t in range(10): F[f'x{t}']=x[:,t]
names=list(F);A=np.stack([F[n] for n in names],1)
for i in range(16):
    y=H[:,i];sel=[];res=[]
    for k in range(5):
        best=None
        for j in range(A.shape[1]):
            if j in sel: continue
            B=A[:,sel+[j]];c,*_=np.linalg.lstsq(B,y,rcond=None);r=1-((y-B@c)**2).mean()/y.var()
            if best is None or r>best[0]: best=(r,j,c)
        sel.append(best[1]);res.append(f"{names[best[1]]}({best[0]:.3f})")
    print(i," ".join(res),np.round(best[2],1))
