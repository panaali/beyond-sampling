import numpy as np
def feats(x):
    N=len(x);F={};pm=np.maximum.accumulate(x,1)
    s=-np.sort(-x,1)
    for t in range(10):
        F[f"x{t}"]=x[:,t]; F[f"pm{t}"]=pm[:,t]; F[f"s{t}"]=s[:,t]
        if t>0:
            F[f"up{t}"]=np.maximum(x[:,t]-pm[:,t-1],0); F[f"dn{t}"]=np.maximum(pm[:,t-1]-x[:,t],0)
    # prefix second max
    ps=np.full(N,-np.inf);m=x[:,0].copy()
    for t in range(1,10):
        ps=np.maximum(ps,np.minimum(m,x[:,t]));m=np.maximum(m,x[:,t]);F[f"ps{t}"]=ps.copy()
    return F
