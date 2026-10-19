import numpy as np
from lib import pf,relufit
x=np.load('x0.npy')[60000:80000];h=np.load('h0.npy')[60000:80000];F=pf(x);one=np.ones(len(x))
P=x[:,:8]; sP=-np.sort(-P,1)
tests={4:[('x9','M8'),('x9','M8','s8'),('x9','M8','mean8')],
       9:[('x8','M7'),('x8','M7','x9'),('x8','M7','s7')],
       5:[('x8','M7'),('x8','M7','x9'),('x8','M7','s7'),('x8','M7','s7','x9'),('x8','mean7','M7','x9')]}
F['s7']=sP[:,1]; F['mean7']=P.mean(1)
for i,ts in tests.items():
    for t in ts:
        A=np.c_[tuple(F[k] for k in t)+(one,)]; c,r=relufit(A,h[:,i]); print(i,t,np.round(c,2),round(r,3))
