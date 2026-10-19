import numpy as np
from scipy.optimize import least_squares
x=np.load('x1.npy')[:40000];h=np.load('h1.npy')[:40000]
o=np.argsort(-x[:,:9],1);s=-np.sort(-x[:,:9],1)
np.set_printoptions(precision=1,suppress=True,linewidth=160)
E=np.eye(9);R=5
X=np.concatenate([E[o[:,r]]*s[:,r:r+1] for r in range(R)]+[x[:,9:10]],1)
for i in [2,9,13,1,5,8,11]:
    hi=h[:,i]
    c,*_=np.linalg.lstsq(X,hi,rcond=None)
    r=least_squares(lambda th:np.maximum(0,X@th)-hi,c)
    print(i,'R2 %.3f'%(1-((np.maximum(0,X@r.x)-hi)**2).mean()/hi.var()),'u %.1f'%r.x[-1])
    print(r.x[:-1].reshape(R,9))
