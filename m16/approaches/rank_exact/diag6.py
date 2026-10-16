"""Limit accuracy of lumped-class canonical-gate surrogates (what the deductive estimator converges to)."""
import sys; sys.path.insert(0,'/home/user/clean-neurons/m16')
import numpy as np
from common import Wi,Wh,Wo
T=10
ref=np.sort(np.random.default_rng(0).standard_normal((1000000,T)),1)[:,::-1]; Kref=(ref>0).sum(1)
EOSK=np.array([ref[Kref==k].mean(0) for k in range(T+1)])
rng=np.random.default_rng(7); N=100000
x=rng.standard_normal((N,T)); order=np.argsort(-x,1); K=(x>0).sum(1); tgt=order[:,1]
def gates(X):
    h=np.zeros((len(X),16)); G=np.zeros((len(X),T,16),bool)
    for t in range(T):
        u=h@Wh.T+X[:,t:t+1]*Wi; G[:,t]=u>0; h=u*G[:,t]
    return G
def surr(X,G):
    h=np.zeros((len(X),16))
    for t in range(T): h=(h@Wh.T+X[:,t:t+1]*Wi)*G[:,t]
    return (h@Wo.T).argmax(1)
def canon(order,K,k_top,arr):
    """top k_top ranks at their true positions; remaining ranks re-assigned to remaining positions:
    arr='pos' -> in increasing position order gets decreasing rank; 'true' -> true ordering."""
    xc=np.empty_like(x)
    if arr=='true':
        np.put_along_axis(xc,order,EOSK[K],1); return xc
    vals=EOSK[K]
    for i in range(len(x)):
        o=order[i]; xc[i,o[:k_top]]=vals[i,:k_top]
        rest=np.sort(o[k_top:]); xc[i,rest]=vals[i,k_top:][::1]  # earliest rest position gets highest rest rank
    return xc
print('true acc',(surr(x,gates(x))==tgt).mean())
for k_top,arr in [(10,'true'),(3,'pos'),(4,'pos'),(2,'pos')]:
    xc=canon(order,K,k_top,arr); print(k_top,arr,'surrogate acc',(surr(x,gates(xc))==tgt).mean())
