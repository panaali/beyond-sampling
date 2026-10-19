import numpy as np
from fitlib import *
x=np.load('x1.npy')[:40000]; H=np.load('H1.npy')[:40000]
d=feats(x)
np.set_printoptions(precision=2,suppress=True,linewidth=220)
F=np.stack([d['P1'],d['P2'],d['P3'],d['x9']],1)
for i in range(16):
  y=H[:,i]
  c,r2,_=relufit(F,y)
  # per prefix-argmax group
  pred=np.zeros_like(y)
  for k in range(9):
    m=d['pa']==k; cc,_,pp=relufit(F[m],y[m]); pred[m]=pp
  r2g=1-((pred-y)**2).mean()/y.var()
  print(i,'global R2 %.3f'%r2,c,' per-pa R2 %.3f'%r2g)
