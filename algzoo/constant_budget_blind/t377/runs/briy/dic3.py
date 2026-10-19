import numpy as np
def pref(x,k):
    """max, second, argmax of x0..xk"""
    p=x[:,:k+1]; o=np.argsort(-p,1); return p.max(1), np.sort(p,1)[:,-2], o[:,0]
def dict3(x):
    D={}
    M=np.maximum.accumulate(x,1)
    for k in range(10): D[f'x{k}']=x[:,k]
    for k in (5,6,7,8):
        Mk,Sk,Ak=pref(x,k); od=(Ak%2==1)
        D[f'M{k}']=Mk; D[f'S{k}']=Sk; D[f'M{k}*odd{k}']=Mk*od; D[f'S{k}*odd{k}']=Sk*od
    for k in range(1,10):
        D[f'relu(M{k-1}-x{k})']=np.maximum(M[:,k-1]-x[:,k],0)
        D[f'relu(x{k}-M{k-1})']=np.maximum(x[:,k]-M[:,k-1],0)
    M8,S8,A8=pref(x,8)
    D['relu(x9-S8)']=np.maximum(x[:,9]-S8,0)
    return D
def greedy(D,y,k=5,names=None):
    names=list(D); A=np.stack([D[n] for n in names],1); sd=A.std(0)+1e-9; A=A/sd
    sel=[]
    act=np.ones(len(y),bool)
    for _ in range(k):
        best=None
        for j in range(A.shape[1]):
            if j in sel: continue
            B=A[:,sel+[j]]; c,*_=np.linalg.lstsq(B,y,rcond=None); r=((B@c-y)**2).mean()
            if best is None or r<best[0]: best=(r,j)
        sel.append(best[1])
    return [names[j] for j in sel]
def relufit(B,y,it=30):
    c,*_=np.linalg.lstsq(B,y,rcond=None)
    for _ in range(it):
        m=(B@c>0)|(y>0)
        c,*_=np.linalg.lstsq(B[m],y[m],rcond=None)
    p=np.maximum(B@c,0); return c,1-((p-y)**2).mean()/y.var()
