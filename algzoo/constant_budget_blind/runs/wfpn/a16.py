import numpy as np
from featlib import basics
np.set_printoptions(suppress=True,precision=3)
H=np.load('Hpre.npy'); x=np.load('xpre.npy'); b=basics(x)
def fitr(B,y):
  m=y>0; c,*_=np.linalg.lstsq(B[m],y[m],rcond=None)
  for it in range(4):
    mm=(B@c>0)|m; c,*_=np.linalg.lstsq(B[mm],y[mm],rcond=None)
  pr=np.maximum(B@c,0); return round(1-((pr-y)**2).mean()/y.var(),3),c
for L in range(3,11):
  t=L-1; y=H[t,:,5]
  r1,_=fitr(np.c_[b['T'][:,t-1],x[:,t]],y)
  r2,c2=fitr(np.c_[b['T'][:,t-1],x[:,t],b['G'][:,1:t].sum(1),x[:,:t].sum(1)],y)
  r3,c3=fitr(np.c_[b['M'][:,t-1],b['T'][:,t-1],x[:,t],b['g'][:,1:t].sum(1),x[:,:t].sum(1)],y)
  print(L,r1,r2,c2,r3,c3)
