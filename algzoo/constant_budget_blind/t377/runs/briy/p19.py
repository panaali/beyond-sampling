import numpy as np
from forms import feats
from dic5 import search
from dic3 import relufit
x=np.load('x1.npy')[:8000]; h=np.load('h1.npy')[:8000]
xt=np.load('x1.npy')[50000:80000]; ht=np.load('h1.npy')[50000:80000]
def ext(x):
    F=feats(x); s=np.sort(x,1)[:,::-1]; o=np.argsort(-x,1)
    for k in range(10):
        F[f's0[am{k}]']=s[:,0]*(o[:,0]==k); F[f's1[a2{k}]']=s[:,1]*(o[:,1]==k)
    F['s0']=s[:,0]; F['s2']=s[:,2]
    return {k:v for k,v in F.items() if not k.startswith('g6')}
D=ext(x); Dt=ext(xt)
for i in [3,9,12,13,15]:
    g=(h[:,i]<=0).mean()>0.05
    n=search(D,h[:,i],5,g); c,_=relufit(np.stack([D[a] for a in n],1),h[:,i])
    p=np.maximum(np.stack([Dt[a] for a in n],1)@c,0); r=1-((p-ht[:,i])**2).mean()/ht[:,i].var()
    print(i,'R2test %.3f'%r,[(a,round(b,2)) for a,b in zip(n,c)],flush=True)
