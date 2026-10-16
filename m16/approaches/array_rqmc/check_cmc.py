"""Validate the exact last-step integration against brute force on x_9."""
import sys, numpy as np
sys.path.insert(0, "m16"); sys.path.insert(0, "m16/approaches/array_rqmc")
from common import Wi, Wh, Wo
from core import init, step, last_step_value, est_mc
import copy
M = (Wi, Wh, Wo); rng = np.random.default_rng(0)
S = init(20); [step(M, S, rng.standard_normal(20)) for _ in range(9)]
v = last_step_value(M, S)
K = 200_000
for i in range(20):
    Si = {k: (np.repeat(val[i:i+1], K, 0) if isinstance(val, np.ndarray) else val) for k, val in S.items()}
    x = rng.standard_normal(K); tgt = np.where(x > Si["m1"], Si["p1"], np.where(x > Si["m2"], 9, Si["p2"]))
    step(M, Si, x); bf = ((Si["h"] @ Wo.T).argmax(1) == tgt).mean()
    print(f"{v[i]:.4f} {bf:.4f} {abs(v[i]-bf)/np.sqrt(max(bf*(1-bf),1e-6)/K):.1f}sd")
print("mean CMC over 400k:", np.mean([est_mc(M, 100_000, np.random.default_rng(s)) for s in range(4)]), "truth 0.953179")
