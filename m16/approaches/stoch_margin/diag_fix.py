"""Diagnostic 3 (random inputs, DIAGNOSIS ONLY): can a small mixture of linearisations fix the top-3 reduction?

Each anchor r puts all 7 low inputs at one value a_r (a quantile of X | X < v3). Its gate pattern gives an exact
linear map L_r(X) = L(a_r) + J_r (X - a_r). Variants, all evaluated on the same low draws as the model (paired),
so each number is the exact-law error of the surrogate (what a perfect deterministic convolution would return):
  lin_mean      : one anchor at the conditional mean (diag_cond 'lin')
  lin_med       : one anchor at the conditional median
  mixR          : equal-weight mixture over R anchors at quantiles (r+0.5)/R            (R = 3, 5)
  nearR         : the anchor whose value is nearest to the mean of the 7 low draws is used for that draw
                  (a partition by one linear statistic, so its law is a computable 2-D law)  (R = 3, 5)
usage: python m16/approaches/stoch_margin/diag_fix.py MODEL [n_skel] [M]
"""
import json
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, "m16/approaches/stoch_margin")
import sm  # noqa: E402

name = sys.argv[1]
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
M = int(sys.argv[3]) if len(sys.argv) > 3 else 256
model = sm.get_models([name])[name]
rng = np.random.default_rng(13)
CH = 400
res = {k: [] for k in ["q", "lin_mean", "lin_med", "mix3", "mix5", "near3", "near5"]}


def lin_surrogate(xb, free, a, low):
    n = len(xb)
    x0 = xb.copy()
    for j in range(7):
        x0[np.arange(n), free[:, j]] = a
    L0 = sm.logits(model, x0)
    J = np.empty((n, 10, 7))
    for j in range(7):
        xe = x0.copy(); xe[np.arange(n), free[:, j]] += 1e-6
        J[:, :, j] = (sm.logits(model, xe) - L0) / 1e-6
    return L0[:, None] + np.einsum("nkj,nmj->nmk", J, low - a[:, None, None])


for c in range(NS // CH):
    pos, val = sm.sample_skeletons(CH, rng)
    v3 = val[:, 2]; P3 = norm.cdf(v3)
    low = sm.trunc_below(v3[:, None, None], rng.uniform(size=(CH, M, 7)))
    x, free = sm.build_inputs(pos, val, low)
    p2 = pos[:, 1]
    ok = sm.logits(model, x.reshape(-1, 10)).reshape(CH, M, 10).argmax(2) == p2[:, None]
    xb = x[:, 0]
    okk = lambda L: L.argmax(2) == p2[:, None]
    res["q"] += list(ok.mean(1))
    res["lin_mean"] += list(okk(lin_surrogate(xb, free, -norm.pdf(v3) / P3, low)).mean(1))
    res["lin_med"] += list(okk(lin_surrogate(xb, free, norm.ppf(0.5 * P3), low)).mean(1))
    xm = low.mean(2)
    for R in (3, 5):
        a = [norm.ppf((r + 0.5) / R * P3) for r in range(R)]
        oks = [okk(lin_surrogate(xb, free, a[r], low)) for r in range(R)]
        res[f"mix{R}"] += list(np.mean([o.mean(1) for o in oks], 0))
        near = np.argmin(np.abs(xm[:, :, None] - np.stack(a, 1)[:, None, :]), 2)
        res[f"near{R}"] += list(np.choose(near, oks).mean(1))
    print(c, flush=True)
q = np.array(res["q"])
out = {"model": name, "NS": NS, "M": M}
for k in res:
    if k != "q":
        d = np.array(res[k]) - q
        out[k] = dict(bias_pp=100 * d.mean(), se_pp=100 * d.std() / np.sqrt(len(d)), mean_abs_pp=100 * np.abs(d).mean())
print(json.dumps(out, indent=1))
json.dump(out, open(f"m16/approaches/stoch_margin/diag_fix_{name}.json", "w"), indent=1)
