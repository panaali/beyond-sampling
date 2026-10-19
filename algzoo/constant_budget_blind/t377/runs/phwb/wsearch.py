import numpy as np, sys, itertools
from lib import relufit
x=np.load('x0.npy')[:15000];h=np.load('h0.npy')[:15000];one=np.ones(len(x))
xt=np.load('x0.npy')[80000:100000];ht=np.load('h0.npy')[80000:100000]
keys=[(a,b) for a in range(10) for b in range(a,10)]
def W(x): return np.stack([x[:,a:b+1].max(1) for a,b in keys],1)
V=W(x); Vt=W(xt)
def ls(A,y,g):
    c,_,_,_=np.linalg.lstsq(A[g],y[g],rcond=None); return 1-((A[g]@c-y[g])**2).mean()/y[g].var()
for i in [int(a) for a in sys.argv[1].split(',')]:
    y=h[:,i]; g=y>0.3; K=len(keys)
    best=max((ls(np.c_[V[:,j],V[:,k],one],y,g),[j,k]) for j in range(K) for k in range(j+1,K))
    sel=best[1]; hist=[]
    for step in range(3):
        cs=[j for j in range(K) if j not in sel]
        A=np.c_[V[:,sel],one]; c,r=relufit(A,y); At=np.c_[Vt[:,sel],np.ones(len(xt))]
        rt=1-((np.maximum(At@c,0)-ht[:,i])**2).mean()/ht[:,i].var()
        print(i,[keys[j] for j in sel],np.round(c,2),'R2test',round(rt,4),flush=True)
        if step==2: break
        s,j=max((ls(np.c_[V[:,sel+[j]],one],y,g),j) for j in cs); sel=sel+[j]
