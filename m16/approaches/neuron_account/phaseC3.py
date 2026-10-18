"""Phase C (v3, review follow-up): recurrent replacement by state injection.

The all-16 replacement of phaseC2.py uses only the t=10 formulas, because every state is then a function of x.
Here the earlier-step formulas carry the computation: set the whole state h_{t0} = ReLU(f_{t0}(x)) (all 16
neurons, phaseC2's validation-chosen family per (j, t0)) and run the TRUE recurrence for steps t0+1..10.
t0 = 10 reproduces phaseC2's all-16 number. Same seeds and sizes as phaseC2 (train 241031, eval 241033); the family
choice is read from results/phaseC2.json, so no validation set is needed.
"""
import json
import numpy as np
from nacommon import sample, run, run_from, target, cases, Wi, Wh, Wo, T, H
from encode import FAMILIES, HFAMILIES, CMP, HingeBank, CellLinear, design, design_h

ALL = FAMILIES + HFAMILIES
labels = [f[0] for f in ALL]
best = json.load(open("results/phaseC2.json"))["best_family"]      # [t-1][j] -> label
Ntr, Nev = 1_000_000, 200_000
xtr, xev = sample(Ntr, 241031), sample(Nev, 241033)
pc = np.concatenate([run(xtr[i:i + 100_000])[1][:, :, CMP] for i in range(0, Ntr, 100_000)])
bank = HingeBank().fit(xtr, pc)
del pc
Htr = np.concatenate([bank.transform(xtr[i:i + 100_000]).astype(np.float32) for i in range(0, Ntr, 100_000)])
Hev = bank.transform(xev).astype(np.float32)
hev, _ = run(xev)
del _
y, ref = target(xev), (hev[:, -1] @ Wo.T).argmax(1)
C = cases(xev)


def dsg(f, x, t, Hx):
    return design(ALL[f], x, t) if f < len(FAMILIES) else design_h(ALL[f], x, t, Hx)


out = {"N": [Ntr, Nev], "base_acc": float((ref == y).mean()), "inject": []}
htr = np.zeros((Ntr, H))
for t in range(1, T + 1):
    ptr = htr @ Wh.T + xtr[:, t - 1:t] * Wi[None]
    F = np.zeros((Nev, H))
    for f in sorted({labels.index(b) for b in best[t - 1]}):
        js = [j for j in range(H) if best[t - 1][j] == labels[f]]
        c, V = dsg(f, xtr, t, Htr)
        mdl = CellLinear().fit(c, V, ptr[:, js])
        del c, V
        c, V = dsg(f, xev, t, Hev)
        F[:, js] = mdl.predict(c, V)
    h = np.maximum(F, 0)
    p = run_from(h, xev, t).argmax(1)
    rec = {"t0": t, "acc": float((p == y).mean()), "agree": float((p == ref).mean()),
           "state_rel_err": float(np.linalg.norm(h - hev[:, t]) / np.linalg.norm(hev[:, t]))}
    for k, m in C.items():
        rec["agree_" + k] = float((p == ref)[m].mean())
    out["inject"].append(rec)
    print(rec, flush=True)
    htr = np.maximum(ptr, 0)
json.dump(out, open("results/phaseC3.json", "w"), indent=1)
