"""T411 diagnostic (after round 1's design failure; design estimates only): step along the pure-W second-order direction
d_W = -M^-1 g_W,S (no E1 term) at eps = .005, .01, .02 -> results/wonly.json. Same design draws as the search."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import design2 as D2  # noqa: E402

Dd = D2.J("dir_r1.json"); GW = D2.J("gW_r1.json"); GE = D2.J("gE_r1.json"); S = np.array(Dd["S"]); H = np.array(Dd["H"])
lam, V = np.linalg.eigh(-H); Minv = (V / np.maximum(lam, .05 * lam.max())) @ V.T
g = np.array(GW["grad"])[S] / GW["P"]; d = -Minv @ g; d /= np.linalg.norm(d)
print("pure-W direction: pred dlogPW/deps %.1f, dacc %.1f eps^2" % (g @ d, d @ H @ d / 2), flush=True)
rows = []
for eps in (0.005, 0.01, 0.02):
    v = D2.W_ORIG.copy(); v[S] += eps * d; m = D2.rnn.unflat(v, D2.SH)
    pw = D2.wline.w(m, D2.T0, 1_000_000, 411003, grad=False); pe = D2.lcmc.e1(m, D2.T0, 5_000_000, 411004, grad=False)
    a, mu, se = D2.D402.acc_plain([D2.rnn.unflat(D2.W_ORIG, D2.SH), m], 10_000_000, 411006)
    rows.append(dict(eps=eps, rW=pw["P"] / GW["P"], rE=pe["P"] / GE["P"], dacc=float(mu[1]), dacc_se=float(se[1]))); print(rows[-1], flush=True)
    D2.J("wonly.json", dict(d=d.tolist(), pred_dlogPW=float(g @ d), pred_dacc_quad=float(d @ H @ d / 2), rows=rows))
