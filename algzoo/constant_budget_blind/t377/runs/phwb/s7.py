import numpy as np
from lib import pf,relufit
x=np.load('x0.npy')[:40000];h=np.load('h0.npy')[:40000];F=pf(x)
E=np.eye(9)[F['p8']]; Q=np.eye(9)[F['q8']]
one=np.ones(len(x))
for i in range(16):
    A1=np.c_[E*F['M8'][:,None],E*F['s8'][:,None],E,F['x9']]
    A2=np.c_[A1,Q*F['s8'][:,None],Q]
    A3=np.c_[A2,E*F['t8'][:,None],F['x8']]
    print(i,[round(relufit(A,h[:,i])[1],3) for A in (A1,A2,A3)])
