import numpy as np
def feats(x):
    N=len(x); s=np.sort(x,1)[:,::-1]; a=np.argsort(-x,1)
    F={}
    for j in range(10): F['x%d'%j]=x[:,j]
    for k in range(4): F['s%d'%k]=s[:,k]
    for j in range(10): F['am%d'%j]=(a[:,0]==j)*1.
    for j in range(10): F['a2%d'%j]=(a[:,1]==j)*1.
    return F
def lin(X,y):
    X1=np.c_[X,np.ones(len(X))]; c=np.linalg.lstsq(X1,y,rcond=None)[0]; r=y-X1@c
    return 1-r.var()/y.var(), c
