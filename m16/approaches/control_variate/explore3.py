import sys, numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16")
from arc_eval import forward, noisy_models, T
from common import Wi, Wh, Wo
from stratified import stratum, N_STRATA
rng=np.random.default_rng(3); n=1_000_000
x=rng.standard_normal((n,T)); tgt=np.argsort(x,1)[:,-2]; s=stratum(x)
P=np.bincount(s,minlength=N_STRATA)/n
print("P",np.round(P,4))
for name,m in [('base',(Wi,Wh,Wo))]+[(f'n05_{i}',m) for i,m in enumerate(noisy_models(6,0.005))]:
    Y=forward(m,x).argmax(1)==tgt
    err=[1-Y[s==h].mean() for h in range(N_STRATA)]
    print(name, np.round(err,4), "err mass", np.round(P*np.array(err),4))
