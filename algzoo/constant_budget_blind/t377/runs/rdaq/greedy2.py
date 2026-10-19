import numpy as np,sys
from feats import *
np.set_printoptions(precision=2,suppress=True)
x=np.load('x1.npy')[:5000];H=np.load('H1.npy')[:5000]
C=catalog(x)
for j in range(10):
    C['rMo%d'%j]=np.maximum(x[:,j]-C['Mo%d'%j],0)
    C['rSo%d'%j]=np.maximum(x[:,j]-C['So%d'%j],0)
    if j>=1: C['rMb%d'%j]=np.maximum(x[:,j]-C['Mb%d'%j],0)
    o9=np.delete(x[:,:9],j,1) if j<9 else x[:,:9]
    s=-np.sort(-o9,1)
    C['Mo9_%d'%j]=s[:,0]; C['So9_%d'%j]=s[:,1]
    C['rMo9_%d'%j]=np.maximum(x[:,j]-s[:,0],0); C['rSo9_%d'%j]=np.maximum(x[:,j]-s[:,1],0)
for i in map(int,sys.argv[1:]):
    for ch,r,c in greedy(C,H[:,i],k=4,it=8):
        pass
    print(i,ch,'%.3f'%r,c,flush=True)
