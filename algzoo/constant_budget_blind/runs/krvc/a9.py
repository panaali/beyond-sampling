import numpy as np, sys
from feats3 import *
from feats2 import greedy
x=np.load('x1.npy')[:6000];H=np.load('H1.npy')[:6000]
F=feats(x); names=list(F); X=np.stack([F[n] for n in names],1)
xv=np.load('x1.npy')[50000:70000];Hv=np.load('H1.npy')[50000:70000]
Fv=feats(xv); Xv=np.stack([Fv[n] for n in names],1)
def rf(cols,y,it=6):
    A=np.c_[X[:,cols],np.ones(len(y))]; m=np.ones(len(y),bool)
    for _ in range(it):
        c,*_=np.linalg.lstsq(A[m],y[m],rcond=None); m2=(y>1e-6)|(A@c>0)
        if (m2==m).all(): break
        m=m2
    p=np.maximum(A@c,0); return 1-((p-y)**2).mean()/y.var(),c
neur=[int(a) for a in sys.argv[1:]] or range(32)
for i in neur:
    y=H[:,i]; cols=[]; hist=[]
    for step in range(4):
        best=None
        for k in range(X.shape[1]):
            if k in cols: continue
            r,c=rf(cols+[k],y)
            if best is None or r>best[0]: best=(r,k,c)
        cols.append(best[1]); hist.append((names[best[1]],round(best[0],3)))
    c=best[2]; A=np.c_[Xv[:,cols],np.ones(len(xv))]; p=np.maximum(A@c,0); yv=Hv[:,i]
    print(i,hist,'val %.3f'%(1-((p-yv)**2).mean()/yv.var()),list(np.round(c,3)),flush=True)
