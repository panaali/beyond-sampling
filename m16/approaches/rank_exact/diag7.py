"""Surrogate limit with classes = (record pattern, K) and gates at the class-conditional mean input."""
import sys; sys.path.insert(0,'/home/user/clean-neurons/m16')
import numpy as np
from diag6 import gates, surr, x, K, tgt, T
def pattern(x):
    N=len(x); code=np.zeros(N,np.int64); m1=x[:,0].copy(); m2=np.full(N,-np.inf)
    for t in range(1,T):
        e=np.where(x[:,t]>m1,0,np.where(x[:,t]>m2,1,2)); code=code*3+e
        m2=np.where(e==0,m1,np.where(e==1,x[:,t],m2)); m1=np.where(e==0,x[:,t],m1)
    return code
ref=np.random.default_rng(99).standard_normal((4000000,T))   # model-free
key=pattern(ref)*11+(ref>0).sum(1)
u,inv,cnt=np.unique(key,return_inverse=True,return_counts=True)
means=np.zeros((len(u),T)); np.add.at(means,inv,ref); means/=cnt[:,None]
print('classes seen',len(u))
kx=pattern(x)*11+K; idx=np.searchsorted(u,kx); ok=(idx<len(u))&(u[np.minimum(idx,len(u)-1)]==kx)
xc=means[np.minimum(idx,len(u)-1)]
print('coverage',ok.mean(),' surrogate acc (pattern,K) gates:',(surr(x,gates(xc))==tgt).mean())
key2=pattern(ref); u2,inv2,c2=np.unique(key2,return_inverse=True,return_counts=True)
m2=np.zeros((len(u2),T)); np.add.at(m2,inv2,ref); m2/=c2[:,None]
xc2=m2[np.searchsorted(u2,pattern(x))]
print('surrogate acc pattern-only gates:',(surr(x,gates(xc2))==tgt).mean())
