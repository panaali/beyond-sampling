import sys, numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16")
from arc_eval import forward, noisy_models, T
from common import Wi, Wh, Wo
from rb9 import Q
rng=np.random.default_rng(7); n=400_000
x=rng.standard_normal((n,T)); tgt=np.argsort(x,1)[:,-2]
for name,m in [('base',(Wi,Wh,Wo)),('n05_1',noisy_models(6,0.005)[1]),('n025_0',noisy_models(6,0.0025)[0])]:
    Y=(forward(m,x).argmax(1)==tgt).astype(float); q=Q(m,x)
    print(name,f"meanY {Y.mean():.5f}+-{Y.std()/np.sqrt(n):.5f} meanQ {q.mean():.5f}  VarY {Y.var():.4f} VarQ {q.var():.4f} ratio {Y.var()/q.var():.2f} corr {np.corrcoef(Y,q)[0,1]:.3f}")
