import numpy as np, itertools
from featlib import *
from feats import fit_relu
x=np.load('x1.npy')[:20000];H=np.load('H1.npy')[:20000]
M=topk(x,9,0);S=topk(x,9,1);a=am9(x);b=a2_9(x)
for i in [1,2,8,15,20,14,18,5,12,19,17,11,30,21]:
    best=[]
    for useA in ['am','a2']:
        v=a if useA=='am' else b
        A=np.eye(9)[v]
        r,c=fit_relu(np.column_stack([M,S,x[:,9],A[:,1:]]),H[:,i],15)
        off=np.r_[0,c[3:11]]; order=np.argsort(off)
        for t in range(1,9):
            for Sset in [order[t:],order[:t]]:
                Sset=sorted(Sset.tolist())
                r2,c2=fit_relu(np.column_stack([M,S,x[:,9],inset(v,Sset)]),H[:,i],15)
                best.append((r2,useA,Sset,np.round(c2,2)))
    best.sort(key=lambda t:-t[0])
    r0,_=fit_relu(np.column_stack([M,S,x[:,9]]),H[:,i],15)
    print(i,'noset %.3f'%r0,' best:',best[0],' 2nd:',best[2][:3])
