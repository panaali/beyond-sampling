import numpy as np, probe
from feat import *
rng=np.random.default_rng(2)
x=rng.standard_normal((40000,10))
H=probe.hidden(x); F=feats(x); names=list(F); A=np.stack([F[n] for n in names],1)
for i in range(32):
    h=H[:,i]; m=h>0
    X=np.c_[A[m],np.ones(m.sum())]
    c,*_=np.linalg.lstsq(X,h[m],rcond=None)
    r=h[m]-X@c; r2=1-r.var()/h[m].var()
    idx=np.argsort(-np.abs(c[:-1]))[:6]
    print(i,"act%.2f R2act %.3f"%(m.mean(),r2)," ".join("%s:%.2f"%(names[j],c[j]) for j in idx),"b%.2f"%c[-1])
