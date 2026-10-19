import numpy as np, itertools
x=np.load('x1.npy')[:50000]; h=np.load('h1.npy')[:50000]
M=np.maximum.accumulate(x,1); m=M[:,9]; s=np.argsort(-x,1); am=s[:,0]
P9=np.maximum(M[:,8]-x[:,9],0); T=m[:,None]-x; G9=np.maximum(x[:,9]-M[:,8],0)
def fit(B,y):
    c,*_=np.linalg.lstsq(B,y,rcond=None); return 1-((y-B@c)**2).mean()/y.var(),c
I=np.eye(10)[am]
for i,f,extra in [(3,P9,None),(6,P9,None),(12,P9,None),(8,T[:,7],None),(14,T[:,4],None),(3,P9,m),(6,P9,m),(12,P9,m),(8,T[:,7],m),(14,T[:,4],m)]:
    y=h[:,i]; best=None
    for bits in range(1,1024):
        S=[j for j in range(10) if bits>>j&1]; ind=I[:,S].sum(1)
        B=np.c_[f,f*ind] if extra is None else np.c_[f,f*ind,extra*ind]
        r,c=fit(B,y)
        if best is None or r>best[0]: best=(r,S,c)
    print(i,'extra' if extra is not None else '', '%.3f'%best[0],best[1],best[2].round(2))
