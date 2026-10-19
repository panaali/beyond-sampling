import numpy as np, probe
from dic4 import dic4
from dic3 import fitrelu
r=lambda z:np.maximum(z,0)
def tf(x):
    D=dic4(x); M8=x[:,:9].max(1); a8=np.argmax(x[:,:9],1)
    D['E8M8']=((a8%2==0)&(a8<8))*M8; D['O8M8']=(a8%2==1)*M8; D['A8M8']=(a8==8)*M8
    D['min(x9,M8)']=np.minimum(x[:,9],M8); D['r(x9-M8)']=r(x[:,9]-M8)
    return D
x=np.random.default_rng(3).standard_normal((12000,10)); h=probe.hidden(x); D=tf(x)
xt=np.random.default_rng(7).standard_normal((40000,10)); ht=probe.hidden(xt); Dt=tf(xt)
base=['E8M8','O8M8','A8M8']
keys=[k for k in D if k not in base]
for i in range(16):
    y=h[:,i]; best=[]
    for k1 in ['x9','min(x9,M8)','r(x9-M8)']:
        for k2 in keys:
            ks=base+[k1,k2]; X=np.stack([D[k] for k in ks],1); b=fitrelu(X,y,10)
            e=((y-r(X@b))**2).mean(); best.append((e,ks,b))
    best.sort(key=lambda t:t[0]); e,ks,b=best[0]
    p=r(np.stack([Dt[k] for k in ks],1)@b); r2=1-((ht[:,i]-p)**2).mean()/ht[:,i].var()
    print(i,ks[3:],'test %.3f'%r2,b.round(1),flush=True)
