import numpy as np
from dic4 import dict4
from dic3 import greedy, relufit
x=np.load('x1.npy')[:10000]; h=np.load('h1.npy')[:10000]
D=dict4(x)
for i in [0,3,4,5,7,8,9,12,13,15]:
    n=greedy(D,h[:,i],5); B=np.stack([D[a] for a in n],1); c,r=relufit(B,h[:,i])
    print(i,'R2relu %.3f'%r,[(a,round(b,1)) for a,b in zip(n,c)])
