import sys, numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16")
from arc_eval import forward, noisy_models, T
from common import Wi, Wh, Wo
base=(Wi,Wh,Wo)
rng=np.random.default_rng(1); n=1_000_000
x=rng.standard_normal((n,T)); tgt=np.argsort(x,1)[:,-2]
Y0=forward(base,x).argmax(1)==tgt
print("base acc",Y0.mean())
for s in (0.0025,0.005):
    for m in noisy_models(6,s):
        Y=forward(m,x).argmax(1)==tgt
        d=(Y!=Y0).mean(); vy=Y.var(); vd=np.var(Y.astype(float)-Y0)
        print(s, f"acc {Y.mean():.4f} disagree {d:.4f} VarY {vy:.4f} Var(Y-Y0) {vd:.4f} ratio {vy/vd:.2f}")
