import sys, numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16")
from arc_eval import forward, noisy_models, T
from common import Wi, Wh, Wo
from rb9 import Q
from basis9 import basis9
rng=np.random.default_rng(9); n=200_000; h=n//2
x=rng.standard_normal((n,T))
for name,m in [('base',(Wi,Wh,Wo)),('n05_1',noisy_models(6,0.005)[1]),('n05_0',noisy_models(6,0.005)[0]),('n025_0',noisy_models(6,0.0025)[0])]:
    q=Q(m,x); out=[name]
    for kind in "AB":
        B=basis9(x,kind)
        c=np.linalg.lstsq(B[:h],q[:h],rcond=None)[0]; r=q[h:]-B[h:]@c
        out.append(f"{kind}({B.shape[1]}): x{q[h:].var()/r.var():.2f}")
    print(" ".join(out),flush=True)
