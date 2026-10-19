import numpy as np
from fitlib import *
x=np.load('x1.npy')[:40000]; H=np.load('H1.npy')[:40000]
d=feats(x)
np.set_printoptions(precision=2,suppress=True,linewidth=220)
M=np.maximum.accumulate(x,1)
for i in [2,9,13]:
  y=H[:,i]
  F=np.column_stack([x,M[:,:9],d['P2'],d['P3']])
  c,r2,_=relufit(F,y)
  print(i,'relu on x0..9,M0..8,P2,P3 R2 %.3f'%r2); print(c)
  # deciles of y vs x2-ish
