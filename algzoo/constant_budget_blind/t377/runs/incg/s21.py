import numpy as np, json, probe
from featlib import F
from rfit import relufit, r2
ch=json.load(open("chosen.json"))
spec={int(i):v[0] for i,v in ch.items()}
spec[7]=["gmax01234567","x9"]; spec[8]=["x9","mex9"]; spec[27]=["x8","gmax01234567"]
spec[25]=["S0","x9","p2nd2"]; spec[16]=["inTop2_7","rec7","mex7"]
alt={10:[["x8","x0","gmax1234567","x9"],["x8","pm5","gmax67","x9"],["x8","x0","gmax1234567","gmax67","x9"]],
     25:[["x9","sm1","p2nd2","pm8"]],8:[["x9","pm8","p2nd8"]],13:[["inTop2_9","inTop2_0","inTop2_1","inTop2_2"]]}
rng=np.random.default_rng(21); xa=rng.standard_normal((40000,10)); 
neg=-np.abs(rng.standard_normal((20000,10)))
xa2=np.r_[xa[:20000],neg]
ha=np.asarray(probe.hidden(xa.astype(np.float32))).astype(float); ha2=np.asarray(probe.hidden(xa2.astype(np.float32))).astype(float)
rng=np.random.default_rng(22); xb=rng.standard_normal((50000,10)); hb=np.asarray(probe.hidden(xb.astype(np.float32))).astype(float)
out={}
for i in range(32):
    cands=[spec[i]]+alt.get(i,[])
    best=None
    for nm in cands:
        X,H=(xa2,ha2) if i in (16,25) else (xa,ha)
        A=np.stack([F(X,n) for n in nm],1); c,_=relufit(A,H[:,i],iters=40)
        B=np.stack([F(xb,n) for n in nm],1); r=r2(hb[:,i],np.maximum(B@c,0))
        print(i,nm,np.round(c,3),"valR2 %.3f"%r, "var %.3g"%hb[:,i].var())
        if best is None or r>best[0]+0.005: best=(r,nm,list(map(float,c)))
    out[i]=best
json.dump(out,open("final_spec.json","w"))
