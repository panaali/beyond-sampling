import numpy as np
from dic import *
x=np.load('x1.npy')[:12000]; h=np.load('h1.npy')[:12000]
D=dict2(x)
for i in range(16):
    n,c,r=greedy(D,h[:,i],5)
    print(i,'R2 %.3f'%r,[(a,round(b,1)) for a,b in zip(n,c)])
