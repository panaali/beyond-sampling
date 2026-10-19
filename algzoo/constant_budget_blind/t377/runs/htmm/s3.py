import numpy as np
from lib import library, fit_relu
x=np.load('x2.npy')[:30000]; h=np.load('h2.npy')[:30000]
F=library(x); names=list(F); X=np.stack([F[k] for k in names],1)
import sys
for i in range(32):
    y=h[:,i]; sel=[]; out=[]
    for step in range(4):
        best=(-9,None,None)
        for j in range(len(names)):
            if j in sel: continue
            b,r=fit_relu(X[:,sel+[j]],y,it=4)
            if r>best[0]: best=(r,j,b)
        sel.append(best[1]); out.append('%s(%.3f)'%(names[best[1]],best[0]))
    b,r=fit_relu(X[:,sel],y,it=10)
    print(i,' '.join(out),'| coefs',np.round(b,2),flush=True)
