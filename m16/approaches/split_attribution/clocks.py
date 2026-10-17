"""T195 alternative (c): label each hidden neuron of zoo(H,T,seed) as clock-like or value-like (SAMPLED diagnostic,
not part of any estimator). 2e5 Gaussian inputs; R^2 of the final h_j regressed on (i) one-hot position of the max
(time since max) and (ii) [x_(1), x_(2), x_(3)] (top order-statistic values, linear + intercept).
clock score = R2_pos - R2_val. usage: python clocks.py H T seed -> clocks/H{H}_T{T}_s{seed}.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "understand_m43"))
import numpy as np
from zoo44 import zoo


def r2(X, y):
    X = np.hstack([X, np.ones((len(X), 1))]); b = np.linalg.lstsq(X, y, rcond=None)[0]
    v = y.var(); return float(1 - ((y - X @ b) ** 2).mean() / v) if v > 0 else 0.0


H, T, seed = (int(v) for v in sys.argv[1:4]); Wi, Wh, Wo = zoo(H, T, seed)
x = np.random.default_rng(7).standard_normal((200000, T)); h = np.zeros((len(x), H))
for t in range(T):
    h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
pos = np.eye(T)[x.argmax(1)][:, 1:]; top = -np.sort(-x, 1)[:, :3]
out = {}
for j in range(H):
    a, b = r2(pos, h[:, j]), r2(top, h[:, j])
    out[j] = dict(r2_pos=a, r2_val=b, clock=a - b, frac_active=float((h[:, j] > 0).mean()))
os.makedirs(os.path.join(HERE, "clocks"), exist_ok=True)
json.dump(out, open(os.path.join(HERE, "clocks", f"H{H}_T{T}_s{seed}.json"), "w"), indent=1)
print({j: (round(v["r2_pos"], 2), round(v["r2_val"], 2)) for j, v in out.items()})
