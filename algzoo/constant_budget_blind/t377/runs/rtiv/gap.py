import numpy as np
from scipy.optimize import least_squares
d=np.load('d1.npz');x=d['x'][:50000];H=d['H'][:50000]
xv=d['x'][50000:];Hv=d['H'][50000:]
def base(x):
    o=np.argsort(-x,1);am=o[:,0];s=-np.sort(-x,1)
    return am,s[:,0],s[:,1]
def design(x,t,par):
    am,M,S=base(x);g=(am%2==par).astype(float)
    return np.c_[g*M,g*S,(1-g)*M,(1-g)*S,x[:,t]]
def fitrelu(X,y,c0):
    r=least_squares(lambda c: np.maximum(X@c,0)-y,c0);return r.x
for i,t in [(3,9),(6,9),(12,9),(8,7),(14,4)]:
    best=None
    for par in [0,1]:
        X=design(x,t,par);c0,*_=np.linalg.lstsq(X,H[:,i],rcond=None)
        c=fitrelu(X,H[:,i],c0)
        Xv=design(xv,t,par);p=np.maximum(Xv@c,0);r=1-((Hv[:,i]-p)**2).mean()/Hv[:,i].var()
        print(i,"t",t,"good parity",par,"R2val %.3f"%r,np.round(c,2))
