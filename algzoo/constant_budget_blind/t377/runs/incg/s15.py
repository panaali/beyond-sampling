import numpy as np, json
from lib import base
from rfit import relufit, r2
d=np.load("d1.npz"); x=d["x"][:10000].astype(float); h=d["h"][:10000].astype(float)
F=base(x)
for j in range(10): F["relu_x%d"%j]=np.maximum(x[:,j],0); F["negx%d"%j]=np.maximum(-x[:,j],0)
for t in range(1,10): F["pmz%d"%t]=np.maximum(x[:,:t+1].max(1),0)
for t in range(1,9): F["smz%d"%t]=np.maximum(x[:,t:].max(1),0)
nm=list(F); M=np.stack([F[n] for n in nm],1)
for i in [29,0,15,5,10,31,26,12,4]:
    y=h[:,i]; sel=[];rs=[]
    for s in range(5):
        best=(-9,None,None)
        for k in range(M.shape[1]):
            if k in sel: continue
            c,r=relufit(M[:,sel+[k]],y,iters=10)
            if r>best[0]: best=(r,k,c)
        sel.append(best[1]); rs.append(round(float(best[0]),3))
    print(i,rs,[nm[k] for k in sel],np.round(best[2],2),flush=True)
