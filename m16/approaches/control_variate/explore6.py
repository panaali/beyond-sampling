import sys, numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16")
from arc_eval import forward, noisy_models, T
from common import Wi, Wh, Wo
from basis import basis
rng=np.random.default_rng(5); n=600_000
x=rng.standard_normal((n,T)); tgt=np.argsort(x,1)[:,-2]
h=n//2
for name,m in [('base',(Wi,Wh,Wo)),('n05_1',noisy_models(6,0.005)[1]),('n05_0',noisy_models(6,0.005)[0]),('n025_0',noisy_models(6,0.0025)[0])]:
    Y=(forward(m,x).argmax(1)==tgt).astype(float); out=[name]
    for kind in "ABC":
        B=basis(x,kind); B=np.hstack([np.ones((n,1)),B])
        c=np.linalg.lstsq(B[:h],Y[:h],rcond=None)[0]; r=Y[h:]-B[h:]@c
        out.append(f"{kind}({B.shape[1]}): x{Y[h:].var()/r.var():.2f}")
    print(" ".join(out),flush=True)
