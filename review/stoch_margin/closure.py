"""T79 variant B: closures in HIDDEN-STATE coordinates given a top-3 skeleton (DIAGNOSIS ONLY, random inputs).
Untried in E31 ('non-Gaussian sequential closure in hidden-state coordinates'). Per skeleton S, truth q from M
paired draws. Reductions (moments taken from particles, i.e. what an exact moment-propagation would give, up to
small particle noise; the closure's fidelity is what is measured, not a cheap implementation):
  once_t   Gaussian closure of the pre-activation z_t at ONE time t (exact mean/cov of z_t), ReLU, then the exact
           network on the true later inputs (paired). t = 0..9. Ceiling for 'Gaussian state at time t'.
  seqG     sequential Gaussian closure: z_t re-Gaussianised (moment-matched) at EVERY step, h_t = relu(z_t).
           Equivalent to assumed-density filtering with a Gaussian on pre-activations (non-Gaussian in h).
  seqMr    sequential Gaussian-MIXTURE closure: at every step the particles are split by the signs of the r most
           ambiguous neurons (closest to P(z>0)=1/2), one Gaussian per cell with its exact weight (<= 2^r comps).
  seqMfull same, split by the full 16-neuron sign pattern of z_t (cells with < 20 particles merged).
usage: python review/stoch_margin/closure.py MODEL [NS] [P]
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import rv  # noqa: E402

name = sys.argv[1]
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
P = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
model = rv.dev_model(name)
Wi, Wh, Wo = model
H = Wh.shape[0]
rng = np.random.default_rng(303)
K, T = 3, rv.T


def gauss_like(Z, rng):
    mu = Z.mean(0)
    C = np.cov(Z.T) if len(Z) > 1 else np.zeros((H, H))
    w, V = np.linalg.eigh(np.atleast_2d(C))
    return mu + rng.standard_normal(Z.shape) @ (V * np.sqrt(np.clip(w, 0, None))).T


def close(Z, mode, rng):
    if mode == "G":
        return gauss_like(Z, rng)
    S = Z > 0
    if mode == "full":
        keys = S @ (1 << np.arange(H))
    else:
        r = int(mode)
        amb = np.argsort(np.abs(S.mean(0) - 0.5))[:r]
        keys = S[:, amb] @ (1 << np.arange(r))
    u, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
    small = cnt[inv] < 20
    out = np.empty_like(Z)
    for g in range(len(u)):
        idx = np.nonzero((inv == g) & ~small)[0]
        if len(idx):
            out[idx] = gauss_like(Z[idx], rng)
    if small.any():
        idx = np.nonzero(small)[0]
        out[idx] = gauss_like(Z[idx], rng) if len(idx) > 1 else Z[idx]
    return out


modes = ["G", "2", "4", "full"]
res = {f"once_{t}": [] for t in range(T)}
res.update({f"seq{m}": [] for m in modes})
res["q"] = []
t0 = time.time()
for s in range(NS):
    pos, val = rv.skeletons(1, K, rng)
    free = rv.free_positions(pos)
    low = rv.low_draws(val[:, -1], P, T - K, rng)
    x = rv.assemble(pos, val, free, low)[0]                   # (P,T)
    lab = pos[0, 1]
    L, pre = rv.run(model, x, keep_pre=True)
    res["q"].append((L.argmax(1) == lab).mean())
    for t in range(T):
        h = np.maximum(gauss_like(pre[:, t], rng), 0)
        for u in range(t + 1, T):
            h = np.maximum(h @ Wh.T + np.outer(x[:, u], Wi), 0)
        res[f"once_{t}"].append(((h @ Wo.T).argmax(1) == lab).mean())
    for m in modes:
        x2 = rv.assemble(pos, val, free, rv.low_draws(val[:, -1], P, T - K, rng))[0]
        h = np.zeros((P, H))
        for u in range(T):
            h = np.maximum(close(h @ Wh.T + np.outer(x2[:, u], Wi), m, rng), 0)
        res[f"seq{m}"].append(((h @ Wo.T).argmax(1) == lab).mean())
    if s % 100 == 0:
        print(s, round(time.time() - t0), flush=True)
q = np.array(res["q"])
out = dict(model=name, NS=NS, P=P, acc=float(q.mean()))
for k, v in res.items():
    if k != "q":
        out[k] = rv.summarize(q, np.array(v), NS)
print(json.dumps(out, indent=1))
json.dump(out, open(f"{__file__.rsplit('/', 1)[0]}/clos_{name}.json", "w"), indent=1)
