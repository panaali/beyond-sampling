import numpy as np
from feats import *
x=np.load('x1.npy')[:40000];H=np.load('H1.npy')[:40000]
d=F(x); q=Q(x)
B1=np.hstack([x,d['P']]); B2=np.hstack([x,d['P'],q]); B3=np.hstack([x,d['P'],q,d['s']])
for i in range(32):
    y=H[:,i]
    print(i,' '.join('%.3f'%fit_relu(B,y)[0] for B in [x,B1,B2,B3]))
