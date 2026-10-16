"""Diagnostic: ADF fidelity per (ordering,K) class and per ordering (Gaussian input with exact order-stat moments)."""
import sys; sys.path.insert(0,'/home/user/clean-neurons/m16')
import numpy as np
from common import Wi,Wh,Wo
from adf import propagate, p_argmax
T=10
def fwd(X):
    h=np.zeros((len(X),16))
    for t in range(T): h=np.maximum(h@Wh.T+X[:,t:t+1]*Wi,0)
    return h@Wo.T
ref=np.sort(np.random.default_rng(0).standard_normal((1000000,T)),1)[:,::-1]
mu=ref.mean(0); S=np.cov(ref.T)
rng=np.random.default_rng(11)
bank=np.sort(rng.standard_normal((20000,T)),1)[:,::-1]
res=[]
for it in range(60):
    perm=rng.permutation(T); tgt=perm[1]
    X=np.empty_like(bank); X[:,perm]=bank
    truth=(fwd(X).argmax(1)==tgt).mean()
    P=np.zeros((T,T)); P[perm,np.arange(T)]=1   # x = P z
    mL,CL=propagate(P@mu,P@S@P.T,Wi,Wh,Wo)
    res.append((truth,p_argmax(mL,CL,tgt)))
r=np.array(res); print('truth, ADF per ordering:',r.mean(0),' mean abs',np.abs(r[:,0]-r[:,1]).mean())
print(np.round(r[:12],3))
