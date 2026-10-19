import numpy as np, probe, sys
from dic5 import dic5
from dic4 import greedyR
from dic3 import fitrelu
def d6(x):
    D=dic5(x); M=np.maximum.accumulate(x,1)
    for t in range(1,10):
        D[f'cap{t}']=np.minimum(x[:,t],M[:,t-1])
        D[f'S{t}']=np.sort(x[:,:t+1],1)[:,-2]
    return D
x=np.random.default_rng(3).standard_normal((6000,10)); h=probe.hidden(x); D=d6(x)
xt=np.random.default_rng(7).standard_normal((30000,10)); ht=probe.hidden(xt); Dt=d6(xt)
for i in [int(a) for a in sys.argv[1].split(',')]:
    g=greedyR(D,h[:,i],5,6); ks=[k for k,_ in g]
    b=fitrelu(np.stack([D[k] for k in ks],1),h[:,i])
    p=np.maximum(np.stack([Dt[k] for k in ks],1)@b,0); r2=1-((ht[:,i]-p)**2).mean()/ht[:,i].var()
    print(i,' '.join(f'{k}({v:.3f})' for k,v in g),'| test %.3f'%r2, b.round(2),flush=True)
