import numpy as np, probe, json
from dic3 import dic3, greedy0, fitrelu
x=np.random.default_rng(3).standard_normal((20000,10)); h=probe.hidden(x); D=dic3(x)
xt=np.random.default_rng(7).standard_normal((50000,10)); ht=probe.hidden(xt); Dt=dic3(xt)
res={}
for i in range(16):
    g=greedy0(D,h[:,i],5); ks=[k for k,_ in g]
    b=fitrelu(np.stack([D[k] for k in ks],1),h[:,i])
    p=np.maximum(np.stack([Dt[k] for k in ks],1)@b,0); r2=1-((ht[:,i]-p)**2).mean()/ht[:,i].var()
    res[i]=(ks,b.tolist())
    print(i,' '.join(f'{k}({v:.3f})' for k,v in g),'| relu test R2 %.3f'%r2, b.round(1),flush=True)
json.dump(res,open('greedy_res.json','w'))
