import numpy as np, probe
from feat import r2; from fit import fitrelu
np.set_printoptions(precision=2,suppress=True,linewidth=160)
rng=np.random.default_rng(9); N=100000
x=rng.standard_normal((N,10)); h=probe.hidden(x)
s=np.sort(x,1)[:,::-1]; o=np.argsort(-x,1); r=np.argsort(o,1); n=np.arange(N)
G=o[:,0]*10+o[:,1]
for i in [0,1,2,3,6,7]:
    y=h[:,i]
    # pair-specific M,S coefs + const, plus shared per-position others
    P=np.zeros((N,100*3)); 
    P[n,G]=s[:,0]; P[n,100+G]=s[:,1]; P[n,200+G]=1
    Oth=x*(r>=2); T3=np.zeros((N,10)); T3[n,o[:,2]]=s[:,2]
    A=np.c_[P,Oth,T3]; c=np.linalg.lstsq(A,y,rcond=None)[0]
    print(i,'pair-table + others lin R2',round(r2(y,A@c),3))
    # residual structure: how much of resid explained by rank-3 value position table?
