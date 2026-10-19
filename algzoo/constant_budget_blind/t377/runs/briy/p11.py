import numpy as np
from dic3 import relufit, pref
x=np.load('x1.npy')[:20000]; h=np.load('h1.npy')[:20000]
P={k:pref(x,k) for k in range(1,10)}
res={}
for i in range(16):
    best=[]
    for k in range(1,10):
        Mk,Sk,Ak=P[k]; od=(Ak%2==1)
        for t in range(10):
            B=np.c_[Mk,Sk,Mk*od,Sk*od,x[:,t]]
            c,r=relufit(B,h[:,i],8); best.append((r,k,t,np.round(c,1)))
    best.sort(key=lambda z:-z[0])
    print(i,[(round(b[0],3),'k',b[1],'t',b[2],list(b[3])) for b in best[:2]])
