import numpy as np
from dic3 import relufit, pref
x=np.load('x1.npy')[:30000]; h=np.load('h1.npy')[:30000]
np.set_printoptions(linewidth=220,suppress=True,precision=1)
M=np.maximum.accumulate(x,1)
S=np.stack([pref(x,k)[1] for k in range(1,10)],1)
for i in [12,15,3,9,4]:
    c,r=relufit(np.c_[x,M[:,1:]],h[:,i]); print(i,'x+M R2 %.3f'%r,c)
    c,r=relufit(np.c_[x,M[:,1:],S],h[:,i]); print(i,'x+M+S R2 %.3f'%r,c)
