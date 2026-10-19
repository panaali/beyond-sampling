import numpy as np
from fitlib import *
x=np.load('x1.npy')[:60000]; H=np.load('H1.npy')[:60000]
d=feats(x)
np.set_printoptions(precision=1,suppress=True,linewidth=220)
F=np.stack([d['P1'],d['P2'],d['P3'],d['x9']],1)
for i in range(16):
  y=H[:,i]; T=[]
  for k in range(9):
    m=d['pa']==k; cc,r2,pp=relufit(F[m],y[m]); T.append(list(cc)+[r2])
  print('neuron',i,'rows P1,P2,P3,x9,R2; cols pa=0..8\n',np.array(T).T)
