import numpy as np
from dic5 import dict5, search
from dic3 import relufit
x=np.load('x1.npy')[:8000]; h=np.load('h1.npy')[:8000]
xt=np.load('x1.npy')[50000:80000]; ht=np.load('h1.npy')[50000:80000]
D=dict5(x); Dt=dict5(xt)
for i in [0,3,4,5,7,8,9,12,13,15]:
    g=(h[:,i]<=0).mean()>0.05
    n=search(D,h[:,i],5,g); c,_=relufit(np.stack([D[a] for a in n],1),h[:,i])
    p=np.maximum(np.stack([Dt[a] for a in n],1)@c,0); r=1-((p-ht[:,i])**2).mean()/ht[:,i].var()
    print(i,'R2test %.3f'%r,[(a,round(b,2)) for a,b in zip(n,c)],flush=True)
