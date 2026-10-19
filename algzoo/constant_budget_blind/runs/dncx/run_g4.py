import numpy as np,sys; from lib3 import library, greedy, R
import lib2
d=np.load('d1.npz'); x=d['x'][:8000].astype(float); h=d['h'][:8000]
F=library(x); F2=lib2.library(x)
for k in ['s0','s1','s2','s3']+['S%d'%j for j in range(10)]+['T%d'%j for j in range(10)]+['gap%d'%t for t in range(1,10)]+['new%d'%t for t in range(1,10)]+['sm%d'%t for t in range(1,10)]: F['L2'+k]=F2[k]
y=x[:,:9]; a=np.argsort(-y,1); C=np.sort(y,1)[:,-3]
for j in range(9): F['C@%d'%j]=C*(a[:,2]==j)
for i in map(int,sys.argv[1:]):
    o,c=greedy(F,h[:,i],5); print(i,o,np.round(c,2),flush=True)
