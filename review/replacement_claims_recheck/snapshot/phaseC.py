"""Phase C: replacement test (causal-scrubbing style).

Every neuron j at every step t is given a fitted formula pre_t[j] ~ f_{j,t}(x) (an encode.FAMILIES member,
fit on a training set to the ORIGINAL network's pre-activations). A replaced neuron's state is
h_t[j] = ReLU(f_{j,t}(x)), computed from the true input; the other neurons run the true recurrence on the
partially replaced state. We replace single neurons, single groups, then groups cumulatively up to all 16.

Formula choice: 'best' = per (j, t) the family with the highest R^2 on a validation set (seed 241032);
'A12' and 'A123' = one fixed family for all (j, t).
Train 1M (seed 241031), validation 200k, evaluation 200k (seed 241033): all disjoint.
Usage: python phaseC.py [Ntrain] [Nval] [Neval]
"""
import json
import os
import sys
import numpy as np
from nacommon import sample, run, target, cases, Wi, Wh, Wo, T, H
from encode import FAMILIES, CellLinear, design, r2

Ntr, Nva, Nev = (int(a) for a in (sys.argv[1:4] if len(sys.argv) > 3 else (1_000_000, 200_000, 200_000)))
xtr, xva, xev = sample(Ntr, 241031), sample(Nva, 241032), sample(Nev, 241033)
hev, _ = run(xev)
FI = list(range(len(FAMILIES)))
Fva = np.zeros((len(FI), T + 1, H))         # validation R^2
KEEP = {"best": None, "A12": 3, "A123": 4}
Fev = {k: np.zeros((T + 1, Nev, H), np.float32) for k in KEEP}   # chosen formula pre-activations on eval
best = np.zeros((T + 1, H), int)
htr, hva, hev_ = np.zeros((Ntr, H)), np.zeros((Nva, H)), np.zeros((Nev, H))
for t in range(1, T + 1):
    ptr = htr @ Wh.T + xtr[:, t - 1:t] * Wi[None]
    pva = hva @ Wh.T + xva[:, t - 1:t] * Wi[None]
    pev_ = hev_ @ Wh.T + xev[:, t - 1:t] * Wi[None]
    allev = []
    for f in FI:
        ctr, Vtr = design(FAMILIES[f], xtr, t)
        mdl = CellLinear().fit(ctr, Vtr, ptr)
        del ctr, Vtr
        c, V = design(FAMILIES[f], xva, t)
        Fva[f, t] = r2(pva, mdl.predict(c, V))
        c, V = design(FAMILIES[f], xev, t)
        allev.append(mdl.predict(c, V).astype(np.float32))
        del mdl
    best[t] = Fva[:, t].argmax(0)
    Fev["best"][t] = np.stack([allev[best[t, j]][:, j] for j in range(H)], 1)
    for k in ("A12", "A123"):
        Fev[k][t] = allev[KEEP[k]]
    del allev
    htr, hva, hev_ = np.maximum(ptr, 0), np.maximum(pva, 0), np.maximum(pev_, 0)
    print("fit t", t, flush=True)
del xtr, htr, ptr

choices = list(KEEP)
y, ref = target(xev), (hev[:, -1] @ Wo.T).argmax(1)
C = cases(xev)


def forward(S, choice):
    F = Fev[choice]
    h = np.zeros((Nev, H))
    for t in range(1, T + 1):
        h = np.maximum(h @ Wh.T + xev[:, t - 1:t] * Wi[None], 0)
        for j in S:
            h[:, j] = np.maximum(F[t][:, j], 0)
    return (h @ Wo.T).argmax(1)


def met(p):
    ok = p == y
    d = {"acc": float(ok.mean()), "agree": float((p == ref).mean())}
    for k, m in C.items():
        d["acc_" + k] = float(ok[m].mean())
        d["agree_" + k] = float((p == ref)[m].mean())
    return d


groups = [("n4", [4]), ("n2", [2]), ("n5,n9", [5, 9]), ("comparators n1,6,7,8", [1, 6, 7, 8]),
          ("n3,n13", [3, 13]), ("clocks n0,10,11,12,14,15", [0, 10, 11, 12, 14, 15])]
out = {"N": [Ntr, Nva, Nev], "base": met(ref), "val_r2": Fva.tolist(),
       "best_family": [[FAMILIES[best[t, j]][0] for j in range(H)] for t in range(1, T + 1)],
       "single": {}, "group": {}, "cumulative": {}}
print("base", out["base"])
for choice in choices:
    out["single"][choice] = {f"n{j}": met(forward([j], choice)) for j in range(H)}
    out["group"][choice] = {g: met(forward(js, choice)) for g, js in groups}
    S, out["cumulative"][choice] = [], []
    for g, js in groups:
        S = S + js
        p = forward(S, choice)
        out["cumulative"][choice].append({"added": g, "S": list(S), **met(p)})
        print(choice, g, round(out["cumulative"][choice][-1]["agree"], 4), round(out["cumulative"][choice][-1]["acc"], 4),
              flush=True)
    if choice == "best":
        p_all = p
    # final-step-only replacement of all 16 (readout part alone)
    hf = np.maximum(Fev[choice][T].astype(float), 0)
    out["cumulative"][choice + "_final_step_only"] = met((hf @ Wo.T).argmax(1))
json.dump(out, open("results/phaseC.json", "w"), indent=1)
os.makedirs("../../../data/t241", exist_ok=True)
np.savez_compressed("../../../data/t241/phaseC_eval.npz", x=xev, ref=ref, p_all=p_all)
