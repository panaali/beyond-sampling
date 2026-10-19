import numpy as np, itertools
from scipy.optimize import least_squares
x=np.load('x1.npy')[:20000];h=np.load('h1.npy')[:20000]
s=-np.sort(-x[:,:9],1);M8,S8,T8=s[:,0],s[:,1],s[:,2];x9=x[:,9]
subsets=[A for r in range(1,9) for A in itertools.combinations(range(9),r)]
for i in [2,9,13,1,5,8,11,14]:
    hi=h[:,i];act=hi>0
    best=[]
    for A in subsets:
        mA=x[:,list(A)].max(1)
        X=np.c_[mA,M8,S8,x9]
        c,res,*_=np.linalg.lstsq(X[act],hi[act],rcond=None)
        e=((X[act]@c-hi[act])**2).sum()
        best.append((e,A))
    best.sort()
    out=[]
    for e,A in best[:5]:
        mA=x[:,list(A)].max(1);X=np.c_[mA,M8,S8,x9]
        c,*_=np.linalg.lstsq(X,hi,rcond=None)
        r=least_squares(lambda th:np.maximum(0,X@th)-hi,c)
        out.append(('%.3f'%(1-((np.maximum(0,X@r.x)-hi)**2).mean()/hi.var()),A,np.round(r.x,1)))
    print(i,out)
