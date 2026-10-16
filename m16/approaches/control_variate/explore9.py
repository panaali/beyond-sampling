import sys, numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16")
from arc_eval import forward, noisy_models, T
from common import Wi, Wh, Wo
from rb9 import Q
rng=np.random.default_rng(10); n=200_000
x=rng.standard_normal((n,T)); tgt=np.argsort(x,1)[:,-2]
for name,m in [('base',(Wi,Wh,Wo)),('n05_1',noisy_models(6,0.005)[1]),('n025_0',noisy_models(6,0.0025)[0])]:
    Y=(forward(m,x).argmax(1)==tgt); q=Q(m,x); qr=Q(m,x,radial=True)
    print(name,f"Y {Y.mean():.5f} Q {q.mean():.5f} Qr {qr.mean():.5f}+-{qr.std()/np.sqrt(n):.5f} VarY/VarQ {Y.var()/q.var():.2f} VarY/VarQr {Y.var()/qr.var():.2f}")
