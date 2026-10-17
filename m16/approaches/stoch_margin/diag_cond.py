"""Diagnostic 1 (random inputs, DIAGNOSIS ONLY): how well do reductions of the low inputs reproduce
q(S) = P(correct | skeleton S), skeleton = positions and values of the top-3 inputs?

For each skeleton S (drawn from the exact law) we draw M sets of 7 low inputs iid N(0,1) | < v3 and compare,
on the SAME draws (paired):
  truth   : the model
  R0      : low inputs at their conditional mean (deterministic summary; H-E10 predicts upward bias)
  lin     : logits linearised in the low inputs at the conditional-mean point (exact Jacobian of its gate pattern)
  add     : cut-HDMR additive surrogate L0 + sum_j [L(base, x_j) - L0] (each 1-D function exact)
  gauss   : Gaussian margin law with the TRUE conditional mean/cov of the 9 margins (2-moment ceiling)
usage: python m16/approaches/stoch_margin/diag_cond.py MODEL [n_skel] [M]
"""
import json
import sys
import time

import numpy as np
from scipy.stats import norm

sys.path.insert(0, "m16/approaches/stoch_margin")
import sm  # noqa: E402

name = sys.argv[1]
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
M = int(sys.argv[3]) if len(sys.argv) > 3 else 256
model = sm.get_models([name])[name]
rng = np.random.default_rng(7)
CH = 400
rows = []
t0 = time.time()
for c in range(NS // CH):
    pos, val = sm.sample_skeletons(CH, rng)
    v3 = val[:, 2]
    low = sm.trunc_below(v3[:, None, None], rng.uniform(size=(CH, M, 7)))
    x, free = sm.build_inputs(pos, val, low)
    p2 = pos[:, 1]
    L = sm.logits(model, x.reshape(-1, 10)).reshape(CH, M, 10)
    ok = L.argmax(2) == p2[:, None]
    mu = -norm.pdf(v3) / norm.cdf(v3)                       # E[X | X < v3]
    xb = x[:, :1].copy()
    for j in range(7):
        xb[np.arange(CH), 0, free[:, j]] = mu
    L0 = sm.logits(model, xb[:, 0])                         # (CH,10)
    ok0 = L0.argmax(1) == p2
    # Jacobian by tiny finite differences (exact inside the gate pattern)
    eps = 1e-6
    J = np.empty((CH, 10, 7))
    for j in range(7):
        xe = xb[:, 0].copy(); xe[np.arange(CH), free[:, j]] += eps
        J[:, :, j] = (sm.logits(model, xe) - L0) / eps
    Llin = L0[:, None] + np.einsum("nkj,nmj->nmk", J, low - mu[:, None, None])
    oklin = Llin.argmax(2) == p2[:, None]
    # cut-HDMR additive surrogate
    Ladd = np.repeat(L0[:, None], M, 1)
    for j in range(7):
        xj = np.repeat(xb, M, 1)
        xj[np.arange(CH), :, free[:, j]] = low[:, :, j]
        Ladd += sm.logits(model, xj.reshape(-1, 10)).reshape(CH, M, 10) - L0[:, None]
    okadd = Ladd.argmax(2) == p2[:, None]
    # Gaussian with true conditional moments of the margins m_k = L_p2 - L_k
    mar = L[np.arange(CH), :, p2][:, :, None] - L
    okg = np.empty(CH)
    for i in range(CH):
        others = [k for k in range(10) if k != p2[i]]
        mm = mar[i][:, others]
        mean, cov = mm.mean(0), np.cov(mm.T) + 1e-9 * np.eye(9)
        z = rng.multivariate_normal(mean, cov, size=2048, method="cholesky")
        okg[i] = (z > 0).all(1).mean()
    for i in range(CH):
        rows.append((val[i, 0], val[i, 1], val[i, 2], ok[i].mean(), float(ok0[i]), oklin[i].mean(), okadd[i].mean(), okg[i]))
    if c % 10 == 0:
        print(c, time.time() - t0, flush=True)

R = np.array(rows)
q = R[:, 3]
out = {"model": name, "NS": NS, "M": M, "acc_from_q": float(q.mean())}
for k, nm in [(4, "R0"), (5, "lin"), (6, "add"), (7, "gauss")]:
    d = R[:, k] - q
    out[nm] = dict(bias_pp=100 * d.mean(), se_pp=100 * d.std() / np.sqrt(len(d)), mean_abs_pp=100 * np.abs(d).mean(),
                   bias_pos_pp=100 * d[R[:, 1] > 0].sum() / len(d), bias_blind_pp=100 * d[R[:, 1] <= 0].sum() / len(d))
print(json.dumps(out, indent=1))
np.save(f"data/stoch_margin/diag_cond_{name}.npy", R)
json.dump(out, open(f"m16/approaches/stoch_margin/diag_cond_{name}.json", "w"), indent=1)
