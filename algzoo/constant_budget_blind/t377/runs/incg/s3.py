import numpy as np, sys
from scipy.optimize import least_squares
from feats import feats
d=np.load("d1.npz"); x=d["x"][:30000].astype(float); h=d["h"][:30000].astype(float)
F=feats(x); names=list(F); A=np.stack([F[n] for n in names],1); A=np.c_[A,np.ones(len(A))]
def r2(y,p): return 1-((y-p)**2).mean()/y.var()
for i in range(32):
    y=h[:,i]
    if y.var()<1e-6: print(i,"dead");continue
    act=y>1e-6
    c0=np.linalg.lstsq(A[act],y[act],rcond=None)[0]
    res=least_squares(lambda c: np.maximum(A@c,0)-y + 1e-3*(A@c<0)*(A@c), c0, max_nfev=200)
    c=res.x; p=np.maximum(A@c,0)
    top=np.argsort(-np.abs(c[:-1]))[:8]
    print(i,"R2 %.3f"%r2(y,p)," ".join("%s:%.2f"%(names[k],c[k]) for k in top),"b:%.2f"%c[-1])
