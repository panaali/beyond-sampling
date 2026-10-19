import numpy as np
x=np.load('x.npy'); h=np.load('h.npy')
M=np.maximum.accumulate(x,1)
srt=-np.sort(-x,1)
def fit(F,y,mask):
  A=np.c_[F[mask],np.ones(mask.sum())]; c,*_=np.linalg.lstsq(A,y[mask],rcond=None)
  r=y[mask]-A@c; return c, 1-r.var()/y[mask].var()
for i in range(8):
  m=h[:,i]>0.5
  c1,r1=fit(x,h[:,i],m); c2,r2=fit(np.c_[x,M],h[:,i],m); c3,r3=fit(np.c_[x,srt],h[:,i],m)
  print(i,"x R2",round(r1,3),"x+M R2",round(r2,3),"x+srt",round(r3,3))
  print("  coef x+M",c2.round(2))
