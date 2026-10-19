import numpy as np
from feats2 import *
x=np.load('x1.npy')[:30000];H=np.load('H1.npy')[:30000]
F=feats(x); names=list(F); X=np.stack([F[n] for n in names],1)
for i in range(32):
    y=H[:,i]; cols,h=greedy(X,y,names,5); r,c=relufit(X,cols,y)
    print(i,h,'relu %.3f'%r,np.round(c,2),flush=True)
