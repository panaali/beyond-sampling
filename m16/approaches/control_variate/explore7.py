import sys, numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16")
from arc_eval import forward, noisy_models, T
from common import Wi, Wh, Wo
from rb9 import Q
rng=np.random.default_rng(8); n=100_000
for name,m in [('base',(Wi,Wh,Wo)),('n05_1',noisy_models(6,0.005)[1]),('n025_0',noisy_models(6,0.0025)[0])]:
    x=rng.standard_normal((n,T)); q1=Q(m,x); out=[f"{name} VarQ {q1.var():.4f}"]
    for k in (8,7,6,5,4,2):   # share x0..x_{k-1}
        y=x.copy(); y[:,k:]=rng.standard_normal((n,T-k)); q2=Q(m,y)
        out.append(f"share{k}: {np.cov(q1,q2)[0,1]/q1.var():.2f}")
    print(" ".join(out),flush=True)
