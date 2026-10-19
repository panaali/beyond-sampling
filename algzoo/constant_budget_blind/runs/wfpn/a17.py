import numpy as np
from featlib import basics
np.set_printoptions(suppress=True,precision=3,linewidth=200)
H=np.load('Hpre.npy'); x=np.load('xpre.npy'); b=basics(x); T=b['T']; N=len(x)
def fitr(B,y):
  m=y>0; c,*_=np.linalg.lstsq(B[m],y[m],rcond=None)
  for it in range(4):
    mm=(B@c>0)|m; c,*_=np.linalg.lstsq(B[mm],y[mm],rcond=None)
  pr=np.maximum(B@c,0); return round(1-((pr-y)**2).mean()/y.var(),3),c
for beta in [1.0,1.22]:
  gp=np.zeros((N,10))
  for t in range(1,10): gp[:,t]=np.maximum(T[:,t-1]-beta*x[:,t],0)
  print("beta",beta)
  for L in [6,8,10]:
    t=L-1; y=H[t,:,3]
    for feats in [[T[:,t],gp[:,t-2],gp[:,t-1],gp[:,t]], [T[:,t],gp[:,t-2],gp[:,t-1],gp[:,t],gp[:,t-4],gp[:,t-3]],[T[:,t],gp[:,t-2],gp[:,t-1],gp[:,t],x[:,t-2],x[:,t-4]]]:
      print(L,*fitr(np.stack(feats,1),y))
  # accumulators: linear in gp history + x
  B=np.c_[gp[:,1:],x]
  print("acc lin R2",[fitr(B,H[9,:,i])[0] for i in range(8)])
