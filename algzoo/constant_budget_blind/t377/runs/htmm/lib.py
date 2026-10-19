import numpy as np
def library(x):
    N=x.shape[0]; F={}
    s=np.sort(x,1)[:,::-1]; o=np.argsort(-x,1)
    P=np.maximum.accumulate(x,1)
    Q=np.full_like(x,-5.)
    for t in range(1,10): Q[:,t]=np.sort(x[:,:t+1],1)[:,-2]
    for t in range(10): F['x%d'%t]=x[:,t]
    F['M']=s[:,0];F['S']=s[:,1];F['T']=s[:,2]
    for t in range(1,9): F['P%d'%t]=P[:,t]
    for t in range(2,9): F['Q%d'%t]=Q[:,t]
    for k in range(1,10):
        F['newmax%d'%k]=np.maximum(x[:,k]-P[:,k-1],0)
    for k in range(2,10):
        F['newsec%d'%k]=np.maximum(x[:,k]-Q[:,k-1],0)
    for k in range(10):
        F['am%d'%k]=(o[:,0]==k)*1.; F['a2%d'%k]=(o[:,1]==k)*1.
    for k in range(10):
        F['x%d-S'%k]=x[:,k]-s[:,1]
    return F
def fit_relu(X,y,it=8):
    X1=np.c_[X,np.ones(len(y))]
    m=y>0.05
    if m.sum()<X1.shape[1]+2: m=np.ones(len(y),bool)
    b=np.linalg.lstsq(X1[m],y[m],rcond=None)[0]
    for _ in range(it):
        pre=X1@b; a=pre>0
        if a.sum()<X1.shape[1]+2: break
        r=y-np.maximum(pre,0)
        d=np.linalg.lstsq(X1[a],r[a],rcond=None)[0]
        bn=b+d
        if ((np.maximum(X1@bn,0)-y)**2).mean()<((np.maximum(pre,0)-y)**2).mean(): b=bn
        else: b=b+0.3*d
    pre=np.maximum(X1@b,0)
    return b,1-((pre-y)**2).mean()/y.var()
