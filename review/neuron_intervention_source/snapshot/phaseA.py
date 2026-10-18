"""Phase A: causal map of M_16,10.

For each neuron j and step t (h_t, t=1..10): zero-, mean- and resample-ablate h_t[j] and run the
rest of the network exactly. Also ablate each neuron (and each group) at every step at once.
Mean = pilot-set mean of h_t[j] (pilot disjoint from eval). Resample = h_t[j] of another random eval input
(a fixed random permutation of the batch; for all-step ablation the donor's clean trajectory).
Outputs: results/phaseA.json (+ npz of the 16x10 matrices).
Usage: python phaseA.py [N]
"""
import json
import sys
import numpy as np
from nacommon import sample, run, run_from, metrics, cases, target, Wi, Wh, Wo, T, H

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
x = sample(N, 241001)
xp = sample(50_000, 241002)
hs, pre = run(x)
hp, _ = run(xp)
mu = hp.mean(0)                      # (T+1, H)
perm = np.random.default_rng(241003).permutation(N)
ref = (hs[:, -1] @ Wo.T).argmax(1)
base = metrics(hs[:, -1] @ Wo.T, x, ref)
print("base", base, flush=True)
C = cases(x)
print({k: float(v.mean()) for k, v in C.items()})

KEYS = list(base.keys())
modes = ["zero", "mean", "resample"]
res = {m: {k: np.zeros((H, T)) for k in KEYS} for m in modes}


def ablated_value(mode, t, j, h):
    if mode == "zero":
        return 0.0
    if mode == "mean":
        return mu[t, j]
    return hs[perm, t, j]


for mode in modes:
    for j in range(H):
        for t in range(1, T + 1):
            h = hs[:, t].copy()
            h[:, j] = ablated_value(mode, t, j, h)
            met = metrics(run_from(h, x, t), x, ref)
            for k in KEYS:
                res[mode][k][j, t - 1] = met[k]
    print(mode, "done", flush=True)


def run_ablate_all_steps(mode, js):
    h = np.zeros((N, H))
    for t in range(T):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
        for j in js:
            h[:, j] = ablated_value(mode, t + 1, j, h)
    return h @ Wo.T


groups = {
    "running_max n2": [2], "record n4": [4], "comparators n1,6,7,8": [1, 6, 7, 8],
    "clocks n0,10,11,12,14,15": [0, 10, 11, 12, 14, 15], "late x8 n5,n9": [5, 9],
    "n3,n13": [3, 13],
}
for j in range(H):
    groups[f"n{j}"] = [j]
allsteps = {}
for g, js in groups.items():
    allsteps[g] = {}
    for mode in modes:
        allsteps[g][mode] = metrics(run_ablate_all_steps(mode, js), x, ref)
        # final-state-only ablation of the group as well
    for mode in modes:
        h = hs[:, -1].copy()
        for j in js:
            h[:, j] = ablated_value(mode, T, j, h)
        allsteps[g][mode + "_final_only"] = metrics(h @ Wo.T, x, ref)
    print(g, {m: round(allsteps[g][m]["acc"], 4) for m in modes}, flush=True)

np.savez("results/phaseA_matrices.npz", **{f"{m}_{k}": res[m][k] for m in modes for k in KEYS})
json.dump({"N": N, "seed_eval": 241001, "seed_pilot": 241002, "base": base,
           "case_shares": {k: float(v.mean()) for k, v in C.items()},
           "matrices": {m: {k: res[m][k].round(5).tolist() for k in KEYS} for m in modes},
           "groups": allsteps}, open("results/phaseA.json", "w"), indent=1, default=float)
