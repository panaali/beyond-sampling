import numpy as np, probe, sys, json
from util import relufit, r2
from featlib import design
C=json.load(open(sys.argv[1]))
rng=np.random.default_rng(11); x=rng.standard_normal((80000,10)); H=probe.hidden(x)
rng=np.random.default_rng(12); xt=rng.standard_normal((80000,10)); Ht=probe.hidden(xt)
out={}
for key,names in C.items():
    i=int(key.split(':')[0]); F=design(x,names); w=relufit(F,H[:,i],40)
    pt=np.maximum(design(xt,names)@w,0); r=r2(pt,Ht[:,i])
    print(key,names,np.round(w,3).tolist(),'testR2',round(r,4),flush=True)
    out[key]=[names,w.tolist(),r]
json.dump(out,open(sys.argv[1].replace('.json','_out.json'),'w'))
