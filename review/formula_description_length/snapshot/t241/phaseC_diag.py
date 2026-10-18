"""Phase C diagnostics.
(1) Precision needed: add i.i.d. Gaussian noise to the TRUE final pre-activations, scaled so that every neuron's
    R^2 = 1 - eps, and measure agreement. Tells how accurate a per-neuron formula must be for 99% agreement.
(2) Variable set vs neuron decomposition: fit the 10 LOGITS directly (A12K-Hall and Hall families at t=10) and
    measure agreement; fit post-ReLU h_10 directly instead of pre_10.
Train 600k (seed 241041), eval 200k (seed 241042).
"""
import json
import numpy as np
from nacommon import sample, run, target, Wi, Wh, Wo, T, H
from encode import HFAMILIES, CMP, HingeBank, CellLinear, design_h, r2

out = {}
xev = sample(200_000, 241042)
hev, pev = run(xev)
ref = (hev[:, -1] @ Wo.T).argmax(1)
y = target(xev)
rng = np.random.default_rng(241043)
sd = pev[:, T].std(0)
out["noise"] = []
for eps in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]:
    p = pev[:, T] + rng.standard_normal(pev[:, T].shape) * sd * np.sqrt(eps)
    pred = (np.maximum(p, 0) @ Wo.T).argmax(1)
    out["noise"].append({"1-R2": eps, "agree": float((pred == ref).mean()), "acc": float((pred == y).mean())})
    print(out["noise"][-1], flush=True)
del hev, pev

xtr = sample(600_000, 241041)
pc = np.concatenate([run(xtr[i:i + 100_000])[1][:, :, CMP] for i in range(0, len(xtr), 100_000)])
bank = HingeBank().fit(xtr, pc)
del pc
Htr = np.concatenate([bank.transform(xtr[i:i + 100_000]).astype(np.float32) for i in range(0, len(xtr), 100_000)])
Hev = bank.transform(xev).astype(np.float32)
htr = np.concatenate([run(xtr[i:i + 100_000])[0][:, -1] for i in range(0, len(xtr), 100_000)])
hevT = run(xev)[0][:, -1]
for fam in (HFAMILIES[2], HFAMILIES[3]):
    c, V = design_h(fam, xtr, T, Htr)
    ce, Ve = design_h(fam, xev, T, Hev)
    lg = CellLinear().fit(c, V, htr @ Wo.T).predict(ce, Ve)
    hh = CellLinear().fit(c, V, htr).predict(ce, Ve)
    rec = {"logit_fit_agree": float((lg.argmax(1) == ref).mean()), "logit_fit_acc": float((lg.argmax(1) == y).mean()),
           "post_relu_fit_agree": float(((hh @ Wo.T).argmax(1) == ref).mean()),
           "post_relu_fit_r2": r2(hevT, hh).tolist()}
    out[fam[0]] = rec
    print(fam[0], {k: v for k, v in rec.items() if k != "post_relu_fit_r2"}, flush=True)
json.dump(out, open("results/phaseC_diag.json", "w"), indent=1)
