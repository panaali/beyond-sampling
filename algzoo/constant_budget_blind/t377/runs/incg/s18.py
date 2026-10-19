import numpy as np
from rfit import relufit, r2
d=np.load("d1.npz"); x=d["x"][:20000].astype(float); h=d["h"][:20000].astype(float)
S=-np.sort(-x,1)
rec=[x[:,0]]+[np.maximum(x[:,t]-x[:,:t].max(1),0) for t in range(1,10)]
# "second-record": running second max increments
p2=[np.sort(x[:,:t+1],1)[:,-2] for t in range(1,10)]
inc2=[p2[0]]+[p2[t]-p2[t-1] for t in range(1,9)]
A=np.stack(rec,1); B=np.stack(inc2,1)
for i in [0,15,4,6,12,19,5,31,10,29,26]:
    y=h[:,i]
    c,r=relufit(np.c_[A,S[:,1],S[:,2]],y,iters=20)
    c2,r2_=relufit(np.c_[A,B],y,iters=20)
    print(i,"rec+S1S2 R2 %.3f"%r,np.round(c,1)); print("   rec+inc2 R2 %.3f"%r2_,np.round(c2,1))
