import numpy as np
def pref(x,k):
    p=x[:,:k+1]; return p.max(1), np.sort(p,1)[:,-2], p.argmax(1)
def feats(x):
    F={}
    M=np.maximum.accumulate(x,1)
    for k in range(10): F[f'x{k}']=x[:,k]
    for k in (6,7,8,9):
        Mk,Sk,Ak=pref(x,k); od=(Ak%2==1).astype(float)
        F[f'M{k}']=Mk; F[f'S{k}']=Sk; F[f'M{k}o']=Mk*od; F[f'S{k}o']=Sk*od
    d=np.diff(np.c_[np.zeros(len(x)),M],axis=1)
    F['Jodd7']=d[:,1]+d[:,3]+d[:,5]+d[:,7]
    for k in range(1,10):
        F[f'dn{k}']=np.maximum(M[:,k-1]-x[:,k],0)   # how far x_k is below running max
        F[f'up{k}']=np.maximum(x[:,k]-M[:,k-1],0)   # how far x_k sets a new running max
        F[f'min{k}']=np.minimum(x[:,k],M[:,k-1])
    F['x456']=x[:,4]+x[:,5]+x[:,6]
    F['x268']=x[:,2]+x[:,6]+x[:,8]
    am=x.argmax(1); g=(am==6).astype(float); s1=np.sort(x,1)[:,-2]
    F['g6x6']=g*x[:,6]; F['g6s1']=g*s1; F['g6x9']=g*x[:,9]; F['s1']=s1
    return F
