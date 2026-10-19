import numpy as np
from dic3 import pref
from dic4 import J
def dict5(x):
    D={}
    M=np.maximum.accumulate(x,1)
    for k in range(10): D[f'x{k}']=x[:,k]
    for k in (6,7,8,9):
        Mk,Sk,Ak=pref(x,k); od=(Ak%2==1)
        D[f'M{k}']=Mk; D[f'S{k}']=Sk; D[f'M{k}*odd{k}']=Mk*od; D[f'S{k}*odd{k}']=Sk*od
        o,e=J(x,k); D[f'Jodd{k}']=o
    for k in range(1,10):
        D[f'relu(M{k-1}-x{k})']=np.maximum(M[:,k-1]-x[:,k],0)
        D[f'relu(x{k}-M{k-1})']=np.maximum(x[:,k]-M[:,k-1],0)
    return D
def score(B,y):
    c,*_=np.linalg.lstsq(B,y,rcond=None)
    for _ in range(4):
        m=(B@c>0)|(y>0); c,*_=np.linalg.lstsq(B[m],y[m],rcond=None)
    return ((np.maximum(B@c,0)-y)**2).mean()
def search(D,y,k=5,gated=False):
    names=list(D); A=np.stack([D[n] for n in names],1)
    sc=(lambda B:score(B,y)) if gated else (lambda B:((B@np.linalg.lstsq(B,y,rcond=None)[0]-y)**2).mean())
    sel=[]
    for _ in range(k):
        r=[(sc(A[:,sel+[j]]),j) for j in range(A.shape[1]) if j not in sel]; sel.append(min(r)[1])
    for sweep in range(2):
        for p in range(k):
            r=[(sc(A[:,sel[:p]+[j]+sel[p+1:]]),j) for j in range(A.shape[1]) if j not in sel or j==sel[p]]
            sel[p]=min(r)[1]
    return [names[j] for j in sel]
