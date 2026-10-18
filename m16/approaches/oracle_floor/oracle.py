"""Oracle floor (DIAGNOSTIC SAMPLING, not an estimator): the E31 'lin' low-input closure of
stoch_margin/diag_cond.py (same construction, same eps=1e-6 finite-difference Jacobian at the conditional-mean
point), run on the DEV2 retrains. Skeletons (top-3 positions+values) from the exact law, M low draws
iid N(0,1)|<v3 each; paired truth q(S) vs lin q_lin(S) on the same draws. add/gauss/R0 of diag_cond are dropped
(R0 is kept, it is free). Per-skeleton rows saved so the bias can be split by region / triple / x9 case.
usage: python oracle.py [NS=20000] [M=256] [first] [last]"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import norm

from panel import HERE, ROOT, dev2, target
sys.path.insert(0, os.path.join(ROOT, "m16/approaches/stoch_margin"))
import sm  # noqa: E402

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
models = [("target", target())] + dev2()
lo = int(sys.argv[3]) if len(sys.argv) > 3 else 0
hi = int(sys.argv[4]) if len(sys.argv) > 4 else len(models)
OUT = os.path.join(HERE, "results/oracle"); os.makedirs(OUT, exist_ok=True)
CH = 200
for k in range(lo, hi):
    name, model = models[k]
    if os.path.exists(os.path.join(OUT, name + ".json")):
        continue
    rng = np.random.default_rng(292100 + k); t0 = time.time(); rows = []
    for c in range(NS // CH):
        pos, val = sm.sample_skeletons(CH, rng)
        v3 = val[:, 2]
        low = sm.trunc_below(v3[:, None, None], rng.uniform(size=(CH, M, 7)))
        x, free = sm.build_inputs(pos, val, low)
        p2 = pos[:, 1]
        L = sm.logits(model, x.reshape(-1, 10)).reshape(CH, M, 10)
        ok = L.argmax(2) == p2[:, None]
        mu = -norm.pdf(v3) / norm.cdf(v3)
        xb = x[:, :1].copy()
        for j in range(7):
            xb[np.arange(CH), 0, free[:, j]] = mu
        L0 = sm.logits(model, xb[:, 0])
        ok0 = L0.argmax(1) == p2
        eps = 1e-6
        J = np.empty((CH, 10, 7))
        for j in range(7):
            xe = xb[:, 0].copy(); xe[np.arange(CH), free[:, j]] += eps
            J[:, :, j] = (sm.logits(model, xe) - L0) / eps
        Llin = L0[:, None] + np.einsum("nkj,nmj->nmk", J, low - mu[:, None, None])
        oklin = Llin.argmax(2) == p2[:, None]
        tri = pos[:, 0] * 100 + pos[:, 1] * 10 + pos[:, 2]
        rows.append(np.c_[val, ok.mean(1), ok0, oklin.mean(1), tri])
    R = np.concatenate(rows)
    np.save(os.path.join(OUT, name + ".npy"), R.astype(np.float32))
    g12, g23 = R[:, 0] - R[:, 1], R[:, 1] - R[:, 2]
    reg = np.where(R[:, 1] <= 0, 0, np.where(np.minimum(g12, g23) >= .05, 3, np.where(g23 <= g12, 1, 2)))
    d = R[:, 5] - R[:, 3]; d0 = R[:, 4] - R[:, 3]
    j = dict(name=name, NS=NS, M=M, seed=292100 + k, acc_from_q=float(R[:, 3].mean()),
             lin=dict(bias_pp=100 * d.mean(), se_pp=100 * d.std() / np.sqrt(len(d)), mean_abs_pp=100 * np.abs(d).mean(),
                      **{f"bias_{nm}_pp": 100 * float(d[reg == i].sum() / len(d)) for i, nm in enumerate(("blind", "band23", "band12", "tail"))}),
             R0=dict(bias_pp=100 * d0.mean(), se_pp=100 * d0.std() / np.sqrt(len(d0))), sec=time.time() - t0)
    json.dump(j, open(os.path.join(OUT, name + ".json"), "w"), indent=1)
    L_ = j["lin"]
    print(k, name, "q %.4f lin bias %+.3f +- %.3f pp (blind %+.3f b23 %+.3f b12 %+.3f tail %+.3f) R0 %+.2f  %.0fs" % (
        j["acc_from_q"], L_["bias_pp"], L_["se_pp"], L_["bias_blind_pp"], L_["bias_band23_pp"], L_["bias_band12_pp"],
        L_["bias_tail_pp"], j["R0"]["bias_pp"], j["sec"]), flush=True)
