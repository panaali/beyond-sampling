import numpy as np
from scipy.optimize import least_squares
x=np.load('x1.npy')[:30000];h=np.load('h1.npy')[:30000]
s=-np.sort(-x,1)
np.set_printoptions(precision=1,suppress=True,linewidth=160)
for i in [1,2,5,8,9,11,13,4,12]:
    hi=h[:,i]
    for name,X in (('x',x),('x,M,S,T',np.c_[x,s[:,:3]])):
        c,*_=np.linalg.lstsq(X,hi,rcond=None)
        r=least_squares(lambda th:np.maximum(0,X@th)-hi,c)
        print(i,name,'R2 %.3f'%(1-((np.maximum(0,X@r.x)-hi)**2).mean()/hi.var()),r.x)
