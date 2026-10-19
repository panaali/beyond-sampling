import numpy as np, json, probe
from featlib import F
from rfit import relufit, r2
sp=json.load(open("final_spec.json"))
rng=np.random.default_rng(31); x=rng.standard_normal((300000,10)); h=np.asarray(probe.hidden(x.astype(np.float32))).astype(float)
xt,ht=x[:200000],h[:200000]; xv,hv=x[200000:],h[200000:]
for i,cands in [(16,[["inTop2_7","rec7","mex7"],["x7","mex7","S0"],["x7","mex7"]]),(25,[["S0","x9","p2nd2"],["S0","x9"],["S0"]])]:
    print(i,"mean",hv[:,i].mean(),"var",hv[:,i].var(),"R2 of zero",r2(hv[:,i],np.zeros(len(hv))))
    for nm in cands:
        A=np.stack([F(xt,n) for n in nm],1); c,_=relufit(A,ht[:,i],iters=60)
        B=np.stack([F(xv,n) for n in nm],1); print("  ",nm,np.round(c,3),"val %.3f"%r2(hv[:,i],np.maximum(B@c,0)))
