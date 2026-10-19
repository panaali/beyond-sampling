import numpy as np, probe, sys
from dic4 import dic4, greedyR
from dic3 import greedy0, fitrelu
x=np.random.default_rng(3).standard_normal((12000,10)); h=probe.hidden(x); D=dic4(x)
xt=np.random.default_rng(7).standard_normal((50000,10)); ht=probe.hidden(xt); Dt=dic4(xt)
mode=sys.argv[1]; neur=[int(a) for a in sys.argv[2].split(',')]
for i in neur:
    g=(greedyR if mode=='R' else greedy0)(D,h[:,i],5); ks=[k for k,_ in g]
    b=fitrelu(np.stack([D[k] for k in ks],1),h[:,i])
    p=np.maximum(np.stack([Dt[k] for k in ks],1)@b,0); r2=1-((ht[:,i]-p)**2).mean()/ht[:,i].var()
    print(i,' '.join(f'{k}({v:.3f})' for k,v in g),'| test %.3f'%r2, b.round(2),flush=True)
