import numpy as np, probe
from fitlib import *
X=np.zeros((10,10))
for p in range(10): X[p,p]=1.0
G=probe.hidden(X)  # G[p,i]
np.save('G.npy',G)
x=np.load('x1.npy')[:30000]; H=np.load('H1.npy')[:30000]
d=feats(x)
for i in range(16):
  y=H[:,i]
  WM=(x[:,:9]*G[:9,i]).max(1)
  WS=(np.maximum(x[:,:9],0)*G[:9,i]).sum(1)
  for nm,W in [('wmax',WM)]:
    F=np.stack([W,x[:,9],d['P1'],d['P2']],1)
    c,r2,_=relufit(F[:,:2],y)
    c2,r22,_=relufit(F,y)
    print(i,nm,'R2 %.3f'%r2,c.round(2),' +P1,P2: %.3f'%r22,c2.round(2))
