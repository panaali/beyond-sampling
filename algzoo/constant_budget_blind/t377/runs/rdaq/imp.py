import numpy as np
from featlib import *
from feats import fit_relu
x=np.load('x1.npy')[:20000];H=np.load('H1.npy')[:20000]
a=am9(x)
for i,j in [(0,2),(4,4),(13,5),(27,0),(7,6),(25,1),(29,1),(28,3)]:
    base=[x[:,j],I1(x,j),I2(x,j)]
    cands={'so':so(x,j),'x9':x[:,9],'J1_9':J1(x,9),'M':topk(x,9,0),'S':topk(x,9,1),'rSo':rSo9(x,j),'xmax-x9':relu(x[:,9]-topk(x,9,0))}
    for k in range(9):
        if k!=j: cands['am=%d'%k]=inset(a,[k]); cands['top2_%d'%k]=top2(x,k)
    res=[]
    for n,z in cands.items():
        r,c=fit_relu(np.column_stack(base+[z]),H[:,i],12); res.append((round(r,3),n,np.round(c,2).tolist()))
    res.sort(key=lambda t:-t[0]); print(i,res[:4])
