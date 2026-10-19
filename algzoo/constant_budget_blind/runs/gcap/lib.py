import numpy as np
x=np.load('x.npy');H=np.load('H.npy')
M=np.maximum.accumulate(x,1)
S=np.zeros_like(x); s=np.full(len(x),-np.inf); m=x[:,0].copy(); S[:,0]=np.nan
for t in range(1,10):
    s=np.maximum(s,np.minimum(m,x[:,t])); m=np.maximum(m,x[:,t]); S[:,t]=s
def r2(p,y): return 1-((y-p)**2).mean()/y.var()
def relufit(F,y,it=10):
    act=y>1e-6
    for _ in range(it):
        c,*_=np.linalg.lstsq(F[act],y[act],rcond=None)
        z=F@c; act=(y>1e-6)|(z>0)
    p=np.maximum(F@c,0); return r2(p,y),c,p
