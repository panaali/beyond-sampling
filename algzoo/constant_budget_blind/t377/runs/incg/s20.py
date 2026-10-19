import numpy as np
from rfit import relufit
d=np.load("d1.npz"); x=d["x"].astype(float); h=d["h"].astype(float)
y=h[:,10]; o=np.argsort(-x,1); S=-np.sort(-x,1)
for t in np.argsort(-y)[:8]: print(round(y[t],2),np.round(x[t],2))
a=y>1e-4
print("argmax|active",np.bincount(o[a,0],minlength=10)," second|active",np.bincount(o[a,1],minlength=10))
m=o[:,0]==8
print("x8 max: frac active",(y[m]>0).mean())
for sp in range(10):
    mm=m&(o[:,1]==sp)
    if mm.sum()>50: print(" second at",sp,"mean %.2f act %.2f"%(y[mm].mean(),(y[mm]>0).mean()))
mm=m
c,r=relufit(np.c_[x[mm,8],S[mm,1],x[mm,9],S[mm,2]],y[mm],iters=30); print("x8max fit x8,S1,x9,S2 %.3f"%r,np.round(c,2))
