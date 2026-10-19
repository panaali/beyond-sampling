import numpy as np, probe, itertools
from feat import *; from fit import *
np.set_printoptions(precision=2,suppress=True,linewidth=160)
rng=np.random.default_rng(5)
N=20000
x=rng.standard_normal((N,10)); h=probe.hidden(x)
s=np.sort(x,1)[:,::-1]; o=np.argsort(-x,1); one=np.ones(N)
for i in [0,1,2,3,6,7]:
    y=h[:,i]; best=[]
    for p,q in itertools.combinations(range(10),2):
        A=np.c_[s[:,0],s[:,1],x[:,p],x[:,q],one]
        c=np.linalg.lstsq(A,y,rcond=None)[0]; best.append((r2(y,A@c),p,q))
    best.sort(reverse=True); r,p,q=best[0]
    A=np.c_[s[:,0],s[:,1],x[:,p],x[:,q],one]; c,rr=fitrelu(A,y)
    A2=np.c_[s[:,0],s[:,1],x[:,p],o[:,0],one]; c2,rr2=fitrelu(A2,y)
    A3=np.c_[s[:,0]+s[:,1],x[:,p],x[:,q],o[:,0],one]; c3,rr3=fitrelu(A3,y)
    print(i,'lin best',[(round(a,3),b,cq) for a,b,cq in best[:3]],'relu',round(rr,3),c,'| x_p+amax',round(rr2,3),c2,'| M+S,xp,xq,amax',round(rr3,3))
