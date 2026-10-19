import numpy as np, probe, sys, json
from dic5 import dic5
from dic4 import greedyR
from dic3 import greedy0, fitrelu
x=np.random.default_rng(3).standard_normal((8000,10)); h=probe.hidden(x); D=dic5(x)
xt=np.random.default_rng(7).standard_normal((40000,10)); ht=probe.hidden(xt); Dt=dic5(xt)
neur=[int(a) for a in sys.argv[1].split(',')]; out={}
for i in neur:
    for mode in ['L','R']:
        g=(greedyR if mode=='R' else greedy0)(D,h[:,i],5); ks=[k for k,_ in g]
        b=fitrelu(np.stack([D[k] for k in ks],1),h[:,i])
        p=np.maximum(np.stack([Dt[k] for k in ks],1)@b,0); r2=1-((ht[:,i]-p)**2).mean()/ht[:,i].var()
        print(i,mode,' '.join(f'{k}({v:.3f})' for k,v in g),'| test %.3f'%r2, b.round(2),flush=True)
        out[f'{i}{mode}']=(ks,b.tolist(),r2)
json.dump(out,open(f'p12_{sys.argv[2]}.json','w'))
