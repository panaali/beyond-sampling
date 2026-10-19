import numpy as np
R=lambda z:np.maximum(z,0)
def library(x):
    N=len(x); s=np.sort(x,1)[:,::-1]; a=np.argsort(-x,1)
    F={}
    for j in range(10): F['x%d'%j]=x[:,j]
    for k in range(4): F['s%d'%k]=s[:,k]
    F['s0-s1']=s[:,0]-s[:,1]; F['s1-s2']=s[:,1]-s[:,2]
    for j in range(10):
        o=np.delete(x,j,1).max(1)
        F['P%d'%j]=R(x[:,j]-o); F['Q%d'%j]=R(o-x[:,j])
        F['am%d'%j]=(a[:,0]==j)*1.; F['a2%d'%j]=(a[:,1]==j)*1.
        F['s0am%d'%j]=s[:,0]*(a[:,0]==j); F['s1a2%d'%j]=s[:,1]*(a[:,1]==j)
        F['R%d'%j]=R(x[:,j]-s[:,1]) ; F['S%d'%j]=R(s[:,1]-x[:,j])
    m=np.maximum.accumulate(x,1)
    for t in range(1,10): F['m%d'%t]=m[:,t]; F['new%d'%t]=R(x[:,t]-m[:,t-1]); F['gap%d'%t]=m[:,t-1]-x[:,t]
    return F
def greedy(F,y,k=4):
    K=list(F); X=np.array([F[q] for q in K]).T; X=(X-X.mean(0))/(X.std(0)+1e-9); yc=y-y.mean()
    sel=[]; out=[]
    for _ in range(k):
        best=None
        for j in range(len(K)):
            if j in sel: continue
            A=X[:,sel+[j]]; c=np.linalg.lstsq(A,yc,rcond=None)[0]; r2=1-((yc-A@c)**2).mean()/yc.var()
            if best is None or r2>best[0]: best=(r2,j)
        sel.append(best[1]); out.append((K[best[1]],round(best[0],3)))
    return out
