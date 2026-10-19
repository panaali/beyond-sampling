import numpy as np, probe, json, sys
from featlib import feature
from dic3 import fitrelu
C=json.load(open('cands.json'))
x=np.random.default_rng(31).standard_normal((40000,10)); h=probe.hidden(x)
xt=np.random.default_rng(21).standard_normal((60000,10)); ht=probe.hidden(xt)
out={}
for i,cl in C.items():
    i=int(i); res=[]
    for ks in cl:
        X=np.stack([feature(k,x) for k in ks],1); b=fitrelu(X,h[:,i],40)
        p=np.maximum(np.stack([feature(k,xt) for k in ks],1)@b,0); r2=1-((ht[:,i]-p)**2).mean()/ht[:,i].var()
        res.append((r2,ks,b.tolist())); print(i,'%.4f'%r2,ks,np.round(b,3),flush=True)
    out[i]=res
json.dump(out,open(sys.argv[1],'w'))
