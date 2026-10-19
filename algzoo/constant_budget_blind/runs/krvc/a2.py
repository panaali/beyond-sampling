import numpy as np
from feats import feats
x=np.load('x1.npy')[:40000];H=np.load('H1.npy')[:40000]
F=feats(x); names=list(F); X=np.stack([F[n] for n in names],1)
def fit(cols,y,mask=None):
    A=np.c_[X[:,cols],np.ones(len(y))]
    m=np.ones(len(y),bool) if mask is None else mask
    c,*_=np.linalg.lstsq(A[m],y[m],rcond=None); return c,A
def relufit(cols,y,it=8):
    m=y>1e-6
    for _ in range(it):
        c,A=fit(cols,y,m); z=A@c; m=(y>1e-6)|(z>0)
    p=np.maximum(A@c,0); return 1-((p-y)**2).mean()/y.var(),c
for i in range(32):
    y=H[:,i]; cols=[]
    out=[]
    for step in range(5):
        best=None
        for k in range(len(names)):
            if k in cols: continue
            c,A=fit(cols+[k],y,y>1e-6) if False else fit(cols+[k],y)
            r=1-((A@c-y)**2).mean()/y.var()
            if best is None or r>best[0]: best=(r,k)
        cols.append(best[1]); out.append('%s'%names[best[1]])
    r,c=relufit(cols,y)
    print(i,out,'lin %.3f relu %.3f'%(best[0],r),np.round(c,2))
