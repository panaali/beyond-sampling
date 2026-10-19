import numpy as np
from dic3 import *
x=np.load('x1.npy')[:15000]; h=np.load('h1.npy')[:15000]
D=dict3(x)
for i in range(16):
    n=greedy(D,h[:,i],5); B=np.stack([D[a] for a in n],1); c,r=relufit(B,h[:,i])
    print(i,'R2relu %.3f'%r,[(a,round(b,1)) for a,b in zip(n,c)])
