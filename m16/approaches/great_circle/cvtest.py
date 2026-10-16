"""Control variates from model-free input features (means known exactly by rotation invariance):
how much do they cut the per-circle variance, vs the same CV applied to plain point sampling?"""
import sys
import numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16"); sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/great_circle")
from common import Wi, Wh, Wo
from arc_eval import forward
from stratified import stratum, N_STRATA

def feats(x):
    s = stratum(x)
    return np.eye(N_STRATA)[s][:, :-1]          # stratum indicators (last dropped)

d = np.load("pilot_0.npz"); f = d["f"]; U, V = d["U"], d["V"]
G = 512; th = np.arange(G) / G * 2 * np.pi
Fc = np.stack([feats((np.cos(th)[:, None] * U[i] + np.sin(th)[:, None] * V[i])).mean(0) for i in range(len(f))])
X = np.c_[np.ones(len(f)), Fc]
beta, *_ = np.linalg.lstsq(X, f, rcond=None)
r = f - X @ beta
print(f"circle: var {f.var():.3e} -> CV residual var {r.var():.3e}  (x{f.var()/r.var():.2f})")
# point sampling with same CV
M = (Wi, Wh, Wo)
x = np.random.default_rng(5).standard_normal((400_000, 10))
y = (forward(M, x).argmax(1) == np.argsort(x, 1)[:, -2]).astype(float)
Xp = np.c_[np.ones(len(y)), feats(x)]
b2, *_ = np.linalg.lstsq(Xp, y, rcond=None); rp = y - Xp @ b2
print(f"points: var {y.var():.3e} -> CV residual var {rp.var():.3e}  (x{y.var()/rp.var():.2f})")
P = d["p"].mean(); PC = d["pc"].mean()
print(f"circle+CV vs points+CV, per unit compute: full {rp.var()/(r.var()*P):.3f}  partial {rp.var()/(r.var()*PC):.3f}")
print(f"circle+CV vs plain points: full {y.var()/(r.var()*P):.3f}  partial {y.var()/(r.var()*PC):.3f}")
