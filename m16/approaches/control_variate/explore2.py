import sys, numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16")
from arc_eval import forward, noisy_models, T
from common import Wi, Wh, Wo
from feats import feats
rng=np.random.default_rng(2); n=2_000_000
x=rng.standard_normal((n,T)); F=feats(x); tgt=F['p2']
nb=len(__import__('feats').GB)+1
keys={
 'stratum-ish(b12,b23,blind)': (F['b12']*nb+F['b23'])*2+F['blind'],
 'p2,p3,b23,blind': ((F['p2']*10+F['p3'])*nb+F['b23'])*2+F['blind'],
 'p1p2p3,b12,b23,blind': (((F['p1']*100+F['p2']*10+F['p3'])*nb+F['b12'])*nb+F['b23'])*2+F['blind'],
 'p1p2p3': (F['p1']*100+F['p2']*10+F['p3']),
}
models=[('base',(Wi,Wh,Wo))]+[(f'n05_{i}',m) for i,m in enumerate(noisy_models(6,0.005))][:3]+[(f'n025_{i}',m) for i,m in enumerate(noisy_models(6,0.0025))][:2]
h=n//2
for name,m in models:
    Y=(forward(m,x).argmax(1)==tgt).astype(float)
    out=[f"{name} acc {Y.mean():.4f}"]
    for kn,k in keys.items():
        k=np.unique(k,return_inverse=True)[1]
        c=np.bincount(k[:h],minlength=k.max()+1); sm=np.bincount(k[:h],Y[:h],minlength=k.max()+1)
        mu=np.where(c>0,sm/np.maximum(c,1),Y[:h].mean())
        r=Y[h:]-mu[k[h:]]
        out.append(f"{kn}: x{Y[h:].var()/r.var():.2f}")
    print(" | ".join(out),flush=True)
