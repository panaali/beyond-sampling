import numpy as np
from feats3 import feats
x=np.load('x1.npy')[:20000];H=np.load('H1.npy')[:20000]
xv=np.load('x1.npy')[60000:80000];Hv=np.load('H1.npy')[60000:80000]
F=feats(x);Fv=feats(xv)
def ev(i,cols,relu=True,show=True):
    y=H[:,i]; A=np.c_[[F[c] for c in cols]+[np.ones(len(y))]].T if cols else np.ones((len(y),1))
    A=np.stack([F[c] for c in cols]+[np.ones(len(y))],1); m=np.ones(len(y),bool)
    for _ in range(15):
        c,*_=np.linalg.lstsq(A[m],y[m],rcond=None)
        if not relu: break
        m2=(y>1e-6)|(A@c>0)
        if (m2==m).all(): break
        m=m2
    Av=np.stack([Fv[cc] for cc in cols]+[np.ones(len(xv))],1); p=Av@c
    if relu: p=np.maximum(p,0)
    yv=Hv[:,i]; r=1-((p-yv)**2).mean()/yv.var()
    if show: print(i,cols,'R2val %.3f'%r,np.round(c,3))
    return r,c
