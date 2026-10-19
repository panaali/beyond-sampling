import numpy as np
from featlib import *
from feats import fit_relu, fit_gated
x=np.load('x1.npy')[:20000];H=np.load('H1.npy')[:20000]
a=am9(x); M=topk(x,9,0); S=topk(x,9,1)
def setsearch(i,base,g,other):
    # offsets by am among gated samples
    A=np.eye(9)[a]
    X=np.column_stack(base+[A[:,k] for k in range(9) if k not in other])
    r,c=fit_gated(X,H[:,i],g>0.5,15) if g is not None else fit_relu(X,H[:,i],15)
    ks=[k for k in range(9) if k not in other]; off=c[len(base):len(base)+len(ks)]
    order=[ks[t] for t in np.argsort(off)]
    best=(-9,)
    for t in range(1,len(order)):
        for Sset in [order[t:],order[:t]]:
            Xs=np.column_stack(base+[inset(a,Sset)])
            r2_,c2=fit_gated(Xs,H[:,i],g>0.5,15) if g is not None else fit_relu(Xs,H[:,i],15)
            if r2_>best[0]: best=(round(r2_,3),sorted(Sset),np.round(c2,2).tolist())
    print(i,'full-offsets %.3f'%r,'best set',best,flush=True)
for i,j in [(29,1),(28,3),(3,8),(10,8),(24,7)]:
    g=1-top2(x,j)
    setsearch(i,[x[:,j],M,S],g,[j])
for i,j in [(7,6),(27,0),(25,1)]:
    setsearch(i,[x[:,j],I1(x,j),I2(x,j)],None,[j])
