import numpy as np
from lib import pf,relufit
x=np.load('x0.npy')[40000:60000];h=np.load('h0.npy')[40000:60000];F=pf(x)
one=np.ones(len(x))
for i in [3,13,5,9,0,15]:
    A=np.c_[x,one]; c,r=relufit(A,h[:,i])
    A2=np.c_[x,F['M8'],F['s8'],one]; c2,r2=relufit(A2,h[:,i])
    print(i,'lin',round(r,3),np.round(c,1),'| +M8,s8',round(r2,3),np.round(c2,1))
