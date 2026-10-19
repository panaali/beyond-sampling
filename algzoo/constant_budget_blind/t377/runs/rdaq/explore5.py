import numpy as np
from feats import *
x=np.load('x1.npy')[:20000];H=np.load('H1.npy')[:20000]
B,names=big(x)
for i in range(32):
    r,c=fit_relu(B,H[:,i],15)
    idx=np.argsort(-np.abs(c[:-1]))[:7]
    print(i,'%.3f'%r,' '.join('%s:%.2f'%(names[k],c[k]) for k in idx),'b=%.2f'%c[-1])
