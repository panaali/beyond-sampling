import numpy as np
from dic import dict2
from dic3 import dict3, greedy, relufit, pref
x=np.load('x1.npy')[:10000]; h=np.load('h1.npy')[:10000]
D=dict2(x); D.update(dict3(x))
M=np.maximum.accumulate(x,1)
for k in range(2,10): D[f'relu(x{k}-S{k-1})']=np.maximum(x[:,k]-pref(x,k-1)[1],0)
for k in range(2,9):
    Mk,Sk,Ak=pref(x,k); D[f'M{k}*odd{k}']=Mk*(Ak%2==1); D[f'M{k}']=Mk
D={k:v for k,v in D.items() if np.isfinite(v).all() and v.std()>0}
print(len(D))
for i in [0,3,4,5,7,8,9,12,13,14,15]:
    n=greedy(D,h[:,i],5); B=np.stack([D[a] for a in n],1); c,r=relufit(B,h[:,i])
    print(i,'R2relu %.3f'%r,[(a,round(b,1)) for a,b in zip(n,c)])
