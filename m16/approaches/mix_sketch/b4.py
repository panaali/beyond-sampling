"""T309 / Fable r3 B4 cheap stop checks (DIAGNOSTIC SAMPLING) on the 80 DEV2 retrains, 1e6 samples each.
(1) readout-gradient stationarity: G = E[(softmax(L) - e_y) h_10^T] (the cross-entropy gradient w.r.t. W_o), against
    the single-batch gradient noise sqrt(tr Var / B); also cos(G, W_o) (an AdamW decoupled-decay signature would be
    G ~ -c W_o in direction for SGD+L2, not for AdamW).
(2) D = E[<p, L>] - E[L_y], correlated with accuracy over the 80 models.  usage: python b4.py -> results/b4.json"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../oracle_floor"))
from panel import ROOT, dev2  # noqa: E402

tr = json.load(open(os.path.join(ROOT, "data/oracle_floor/dev2_truths.json")))["truths"]
N, CH = 1_000_000, 250_000
rows = []
for k, (name, (Wi, Wh, Wo)) in enumerate(dev2()):
    rng = np.random.default_rng(309500 + k)
    S1 = np.zeros((10, 16)); S2 = np.zeros((10, 16)); D = 0.0; ce = 0.0
    for _ in range(N // CH):
        x = rng.standard_normal((CH, 10)); y = np.argsort(x, 1)[:, -2]
        h = np.zeros((CH, 16))
        for t in range(10):
            h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
        L = h @ Wo.T; Lm = L - L.max(1, keepdims=True); p = np.exp(Lm); p /= p.sum(1, keepdims=True)
        r = p.copy(); r[np.arange(CH), y] -= 1
        g = r[:, :, None] * h[:, None, :]                                  # per-sample gradient (CH,10,16)
        S1 += g.sum(0); S2 += (g ** 2).sum(0)
        D += ((p * L).sum(1) - L[np.arange(CH), y]).sum(); ce += -np.log(p[np.arange(CH), y] + 1e-300).sum()
    G = S1 / N; var = S2 / N - G ** 2; trv = var.sum()
    nG = np.linalg.norm(G)
    rows.append(dict(name=name, acc=tr[name]["acc"], G_norm=float(nG), G_se=float(np.sqrt(trv / N)),
                     ratio_B1024=float(nG / np.sqrt(trv / 1024)), ratio_B256=float(nG / np.sqrt(trv / 256)),
                     ratio_B4096=float(nG / np.sqrt(trv / 4096)), cos_G_Wo=float((G * Wo).sum() / nG / np.linalg.norm(Wo)),
                     D=float(D / N), ce=float(ce / N)))
    print(k, rows[-1], flush=True)
a = np.array([r["acc"] for r in rows]); Dv = np.array([r["D"] for r in rows]); ce = np.array([r["ce"] for r in rows])
R = np.array([r["ratio_B1024"] for r in rows])
out = dict(rows=rows, ratio_B1024=dict(median=float(np.median(R)), min=float(R.min()), max=float(R.max()),
           frac_gt_010=float((R > .1).mean())), G_over_se_median=float(np.median([r["G_norm"] / r["G_se"] for r in rows])),
           cos_G_Wo_median=float(np.median([r["cos_G_Wo"] for r in rows])),
           corr_D_acc=float(np.corrcoef(Dv, a)[0, 1]), corr_ce_acc=float(np.corrcoef(ce, a)[0, 1]),
           D_mean=float(Dv.mean()), D_sd=float(Dv.std(ddof=1)))
json.dump(out, open(os.path.join(HERE, "results/b4.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
