"""Paired comparison across models on the SAME surface points (seed 2026): how much of delta / mass is shared?
Also the margin decomposition m = tie-pair + max + rest per model (cancellation factor)."""
import numpy as np
import feats as F
names = ["orig"] + [f"seed{s}" for s in range(5)] + [f"noisy0.0025_{k}" for k in range(3)] + [f"noisy0.005_{k}" for k in range(3)]
D = {}
import os, wtool as W
if not os.path.exists("data/orig30k.npz"):
    np.savez_compressed("data/orig30k.npz", **W.build(W.get_model("orig"), 30000, seed=2026))
for nm in names:
    R = dict(np.load(f"data/{'orig30k' if nm == 'orig' else nm}.npz")); D[nm] = R
keep = D["orig"]["v"] > 0
o = D["orig"]
print("model           E[mass]  E|delta|  sd(delta)  corr(delta,orig) corr(mass,orig) | sd(m) sd(pair) sd(max) sd(rest) cancel=sqrt(sum var)/sd(m) | R2 of m by max-term alone")
for nm in names:
    R = D[nm]; k = keep
    d = np.clip(R["delta"][k], -.3, .3); m_ = R["mass"][k]
    y, J = R["y"][k], R["Jab"][k]; r = np.arange(k.sum()); a, b, p1 = R["a"][k], R["b"][k], R["p1"][k]
    c = J * y; m = c.sum(1); pr = c[r, a] + c[r, b]; mx = c[r, p1]; rest = m - pr - mx
    canc = np.sqrt(pr.var() + mx.var() + rest.var()) / m.std()
    r2mx = np.corrcoef(m, mx)[0, 1] ** 2
    print(f"{nm:15s} {m_.mean():.4f}  {np.abs(d).mean():.4f}    {d.std():.4f}     {np.corrcoef(d, np.clip(o['delta'][k],-.3,.3))[0,1]:+.3f}           {np.corrcoef(m_, o['mass'][k])[0,1]:+.3f}       | {m.std():.2f} {pr.std():.1f} {mx.std():.1f} {rest.std():.1f} {canc:.1f} | {r2mx:.3f}")
