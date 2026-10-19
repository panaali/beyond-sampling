import numpy as np
R=lambda z:np.maximum(z,0)
def library(x):
    N=len(x); s=np.sort(x,1)[:,::-1]; a=np.argsort(-x,1)
    F={}
    for j in range(10): F['x%d'%j]=x[:,j]
    for k in range(4): F['s%d'%k]=s[:,k]
    for j in range(10):
        o=np.delete(x,j,1).max(1)
        F['P%d'%j]=R(x[:,j]-o); F['Q%d'%j]=R(o-x[:,j])
        F['R%d'%j]=R(x[:,j]-s[:,1]); F['S%d'%j]=R(s[:,1]-x[:,j]); F['T%d'%j]=R(s[:,2]-x[:,j])
        F['s0am%d'%j]=s[:,0]*(a[:,0]==j); F['s1a2%d'%j]=s[:,1]*(a[:,1]==j)
        F['s0am<=%d'%j]=s[:,0]*(a[:,0]<=j); F['s1a2<=%d'%j]=s[:,1]*(a[:,1]<=j)
    m=np.maximum.accumulate(x,1); sm=np.maximum.accumulate(x[:,::-1],1)[:,::-1]
    for t in range(1,10):
        F['m%d'%t]=m[:,t]; F['new%d'%t]=R(x[:,t]-m[:,t-1]); F['gap%d'%t]=R(m[:,t-1]-x[:,t]); F['sm%d'%t]=sm[:,t]
        F['m2_%d'%t]=np.sort(x[:,:t+1],1)[:,-2]
    return F
def fitrelu(X,y,it=6):
    act=y>0
    if act.sum()<10: act=np.ones(len(y),bool)
    c=np.linalg.lstsq(X[act],y[act],rcond=None)[0]
    for _ in range(it):
        p=X@c; act=(p>0)|(y>0)
        c=np.linalg.lstsq(X[act],y[act],rcond=None)[0]
    p=R(X@c); return 1-((y-p)**2).mean()/y.var(), c
def greedy(F,y,k=4,K=None):
    K=K or list(F); sel=[]; out=[]
    for _ in range(k):
        best=None
        for q in K:
            if q in sel: continue
            X=np.array([F[z] for z in sel+[q]]).T
            r2,c=fitrelu(X,y,3)
            if best is None or r2>best[0]: best=(r2,q,c)
        sel.append(best[1]); out.append((best[1],round(best[0],3)))
    return out,best[2]
