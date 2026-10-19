import numpy as np
from featlib import basics, design
H=np.load('Hpre.npy'); x=np.load('xpre.npy'); b=basics(x)
def fitr(B,y):
  m=y>0; c,*_=np.linalg.lstsq(B[m],y[m],rcond=None)
  for it in range(4):
    mm=(B@c>0)|m; c,*_=np.linalg.lstsq(B[mm],y[mm],rcond=None)
  pr=np.maximum(B@c,0); return round(1-((pr-y)**2).mean()/y.var(),3),c.round(2)
for i in [0,1,2,6,7]:
  B=np.stack(design(i,b),1)
  print(i, fitr(B,H[9,:,i])[0], "+true n3 final", fitr(np.c_[B,H[9,:,3]],H[9,:,i]), "+n3 hist", fitr(np.c_[B,H[:,:,3].T],H[9,:,i])[0])
