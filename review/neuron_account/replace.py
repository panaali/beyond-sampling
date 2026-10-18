"""Review of T241 claims C1 (all-16 replacement agreement .9554), single-neuron replacement, D1 (near-tie/blind share
of disagreements) and the held-out discipline. Own features/fits (rv.py), own seeds and RNG (Philox 11/12/13).
usage: python replace.py [Ntrain Nval Neval] -> results/replace.json, data/replace_eval.npz"""
import json
import os
import sys
import time

import numpy as np

import rv

Ntr, Nva, Nev = (int(v) for v in sys.argv[1:4]) if len(sys.argv) > 3 else (1_000_000, 200_000, 200_000)
t0 = time.time()
X = {"tr": rv.gauss(Ntr, 11), "va": rv.gauss(Nva, 12), "ev": rv.gauss(Nev, 13)}
# held-out discipline: no row shared between the three sets (hash of exact bytes)
hs = {k: set(map(hash, map(bytes, v.view(np.uint8).reshape(len(v), -1)))) for k, v in X.items()}
overlap = {f"{a}&{b}": len(hs[a] & hs[b]) for a, b in (("tr", "va"), ("tr", "ev"), ("va", "ev"))}
del hs
PRE = {k: rv.states(v, np.float32)[0] for k, v in X.items()}
print("overlap", overlap, "states", round(time.time() - t0), flush=True)

# comparator hinge bank: c_{k,s} = ReLU(A123K formula of comparator k at step s), fit on train only
A123K = rv.FAMS[3]
HB = {k: np.zeros((len(v), rv.T + 1, 4), np.float32) for k, v in X.items()}
for s in range(1, rv.T + 1):
    c, V = rv.design(A123K, X["tr"], s)
    mdl = rv.CellRidge().fit(c, V, PRE["tr"][:, s][:, rv.CMP].astype(np.float64))
    for k in X:
        c, V = rv.design(A123K, X[k], s)
        HB[k][:, s] = np.maximum(mdl.predict(c, V), 0)
print("hinges", round(time.time() - t0), flush=True)

nF = len(rv.FAMS)
R2va = np.zeros((nF, rv.T + 1, rv.H)); R2ev = np.zeros((nF, rv.T + 1, rv.H))
F = {"best": np.zeros((rv.T + 1, Nev, rv.H), np.float32), "best_noH": np.zeros((rv.T + 1, Nev, rv.H), np.float32)}
choice = np.zeros((rv.T + 1, rv.H), int)
for t in range(1, rv.T + 1):
    ev_pred = []
    for f, fam in enumerate(rv.FAMS):
        c, V = rv.design(fam, X["tr"], t, HB["tr"])
        mdl = rv.CellRidge().fit(c, V, PRE["tr"][:, t].astype(np.float64)); del c, V
        c, V = rv.design(fam, X["va"], t, HB["va"]); R2va[f, t] = rv.r2(PRE["va"][:, t], mdl.predict(c, V))
        c, V = rv.design(fam, X["ev"], t, HB["ev"]); p = mdl.predict(c, V)
        R2ev[f, t] = rv.r2(PRE["ev"][:, t], p); ev_pred.append(p.astype(np.float32)); del mdl, c, V
    choice[t] = R2va[:, t].argmax(0)                       # chosen on VALIDATION
    F["best"][t] = np.stack([ev_pred[choice[t, j]][:, j] for j in range(rv.H)], 1)
    nh = R2va[:4, t].argmax(0)
    F["best_noH"][t] = np.stack([ev_pred[nh[j]][:, j] for j in range(rv.H)], 1)
    print("t", t, "min chosen eval R2", R2ev[choice[t], t, np.arange(rv.H)].min().round(4), round(time.time() - t0), flush=True)

xe = X["ev"]; y = rv.answer(xe)
_, he = rv.states(xe); ref = (he[:, rv.T] @ rv.Wo.T).argmax(1)


def run(S, which="best"):
    h = np.zeros((Nev, rv.H))
    for t in range(1, rv.T + 1):
        h = np.maximum(h @ rv.Wh.T + np.outer(xe[:, t - 1], rv.Wi), 0)
        if S:
            h[:, S] = np.maximum(F[which][t][:, S], 0)
    return (h @ rv.Wo.T).argmax(1)


def met(p):
    return {"agree": float((p == ref).mean()), "acc": float((p == y).mean())}


out = {"N": [Ntr, Nva, Nev], "overlap": overlap, "net_acc": float((ref == y).mean()),
       "families": [f[0] for f in rv.FAMS], "chosen_t10": [rv.FAMS[i][0] for i in choice[rv.T]],
       "eval_r2_t10_chosen": R2ev[choice[rv.T], rv.T, np.arange(rv.H)].tolist(),
       "eval_r2_t10_by_family": {fam[0]: R2ev[f, rv.T].tolist() for f, fam in enumerate(rv.FAMS)},
       "single": {f"n{j}": met(run([j])) for j in range(rv.H)}}
groups = [[4], [2], [5, 9], [1, 6, 7, 8], [3, 13], [0, 10, 11, 12, 14, 15]]
S = []; out["cumulative"] = []
for g in groups:
    S = S + g; out["cumulative"].append({"S": list(S), **met(run(S))})
p_all = run(list(range(rv.H))); out["all16"] = met(p_all); out["all16_noH"] = met(run(list(range(rv.H)), "best_noH"))
# D1: disagreements near ties / blind
s = -np.sort(-xe, 1); g12, g23 = s[:, 0] - s[:, 1], s[:, 1] - s[:, 2]; blind = (xe > 0).sum(1) <= 1
near = (g12 < 0.05) | (g23 < 0.05) | blind; dis = p_all != ref
out["D1"] = {"mass_near": float(near.mean()), "share_dis_near": float(dis[near].sum() / dis.sum()),
             "dis_rate_near": float(dis[near].mean()), "dis_rate_far": float(dis[~near].mean()),
             "dis_on_net_errors": float(dis[ref != y].mean()), "dis_on_net_correct": float(dis[ref == y].mean()),
             "blind_mass": float(blind.mean()), "blind_dis_rate": float(dis[blind].mean())}
out["sec"] = time.time() - t0
os.makedirs(os.path.join(rv.HERE, "results"), exist_ok=True)
json.dump(out, open(os.path.join(rv.HERE, "results", "replace.json"), "w"), indent=1)
np.savez_compressed(os.path.join(rv.HERE, "data", "replace_eval.npz"), x=xe, ref=ref, p_all=p_all)
print(json.dumps({k: v for k, v in out.items() if k not in ("eval_r2_t10_by_family",)}), flush=True)
