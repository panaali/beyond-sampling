import numpy as np, probe
from featlib import F
from rfit import relufit, r2
rng=np.random.default_rng(41); x=rng.standard_normal((300000,10)); h=np.asarray(probe.hidden(x.astype(np.float32))).astype(float)[:,16]
neg=-np.abs(rng.standard_normal((40000,10))); hn=np.asarray(probe.hidden(neg.astype(np.float32))).astype(float)[:,16]
for nm in [["isMax7","x7"],["isMax7","x7","x9"],["isMax7","S0","x9"],["inTop2_7","S0","x9"],["isMax7","inTop2_7","S0"]]:
    A=np.stack([F(neg,n) for n in nm],1); c,rn=relufit(A,hn,iters=60)
    B=np.stack([F(x,n) for n in nm],1); print(nm,np.round(c,3),"neg R2 %.3f"%rn,"gauss R2 %.3f"%r2(h,np.maximum(B@c,0)))
