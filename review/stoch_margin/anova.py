"""T79 variant A: ANOVA (conditional-expectation) additive surrogate of the logits given a top-K skeleton
(DIAGNOSIS ONLY, random inputs; ORACLE for the components).
E31 tested cut-HDMR additivity (terms anchored at the mean point; -3.5..-3.8 pp at K=3). E26 found an oracle
ANOVA-additive fit far better than cut-HDMR. Here, per skeleton S:
  g_j(x) = E[L | x_j = x, S]  (vector of 10 logits) on a G-node grid of conditional quantiles of x_j, each node by
           an inner MC average over NI completions of the other low inputs (common random numbers across nodes),
  L_A(x) = E[L|S] + sum_j (g_j(x_j) - E[L|S]),  g_j interpolated linearly in the conditional-CDF coordinate.
Its law is that of a sum of independent terms (exactly computable by convolution / characteristic functions, so a
deterministic route exists ONCE the g_j are known). Evaluated on M paired outer draws, like E31.
  anova      L_A on the outer draws (oracle, NI inner draws; inner MC noise biases it DOWN, so also reported at NI/4
             and Richardson-extrapolated to NI = infinity)
  anova_r    L_A + Gaussian residual with the true residual covariance (oracle 'additive + noise' law)
  anova_m    ANOVA additive model of the 9 MARGINS m_k = L_p2 - L_k (same thing: margins are linear in L) - check
usage: python review/stoch_margin/anova.py MODEL K [NS] [M] [NI] [G]
"""
import json
import sys
import time

import numpy as np
from scipy.stats import norm

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import rv  # noqa: E402

name, K = sys.argv[1], int(sys.argv[2])
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
M = int(sys.argv[4]) if len(sys.argv) > 4 else 256
NI = int(sys.argv[5]) if len(sys.argv) > 5 else 256
G = int(sys.argv[6]) if len(sys.argv) > 6 else 24
model = rv.dev_model(name)
rng = np.random.default_rng(404)
nl = rv.T - K
ugrid = (np.arange(G) + 0.5) / G
res = {k: [] for k in ["q", "anova", "anova_q", "anova_r", "lin_ref"]}
t0 = time.time()


def interp_terms(gj, uo):
    """gj (G,10) on ugrid; uo (M,) -> (M,10), linear interpolation with flat extrapolation."""
    return np.stack([np.interp(uo, ugrid, gj[:, k]) for k in range(10)], 1)


for s in range(NS):
    pos, val = rv.skeletons(1, K, rng)
    vK = val[:, -1]
    free = rv.free_positions(pos)
    low = rv.low_draws(vK, M, nl, rng)
    x = rv.assemble(pos, val, free, low)[0]
    lab = pos[0, 1]
    L = rv.run(model, x)[0]
    res["q"].append((L.argmax(1) == lab).mean())
    inner = rv.low_draws(vK, NI, nl, rng)
    xin = rv.assemble(pos, val, free, inner)[0]                  # (NI,T)
    Lin = rv.run(model, xin)[0]
    nodes = norm.ppf(ugrid * norm.cdf(vK[0]))                     # conditional quantiles of x_j
    big = np.repeat(xin[None, None], nl, 0).repeat(G, 1)          # (nl,G,NI,T)
    for j in range(nl):
        big[j, :, :, free[0, j]] = nodes[:, None]
    Lg = rv.run(model, big.reshape(-1, rv.T))[0].reshape(nl, G, NI, 10)
    uo = norm.cdf(low[0]) / norm.cdf(vK[0])                       # (M,nl) conditional-CDF coordinates
    for tag, n_in in (("anova", NI), ("anova_q", NI // 4)):
        E0 = Lin[:n_in].mean(0)
        gj = Lg[:, :, :n_in].mean(2)                              # (nl,G,10)
        LA = E0 + sum(interp_terms(gj[j], uo[:, j]) - E0 for j in range(nl))
        res[tag].append((LA.argmax(1) == lab).mean())
        if tag == "anova":
            R = L - LA
            Rc = R - R.mean(0)
            w, V = np.linalg.eigh(Rc.T @ Rc / (M - 1))
            LAr = LA + R.mean(0) + rng.standard_normal((M, 10)) @ (V * np.sqrt(np.clip(w, 0, None))).T
            res["anova_r"].append((LAr.argmax(1) == lab).mean())
    # reference: lin on the same skeleton and draws (re-check of E31's best reduction)
    mu = rv.cond_mean(vK)
    xb = rv.assemble(pos, val, free, np.full((1, 1, nl), mu[0]))[:, 0]
    L0, pre0 = rv.run(model, xb, keep_pre=True)
    Jl = rv.jacobian(model, pre0)[0][:, free[0]]
    res["lin_ref"].append(((L0 + (low[0] - mu[0]) @ Jl.T).argmax(1) == lab).mean())
    if s % 100 == 0:
        print(s, round(time.time() - t0), flush=True)
q = np.array(res["q"])
out = dict(model=name, K=K, NS=NS, M=M, NI=NI, G=G, acc=float(q.mean()))
for k in ["anova", "anova_q", "anova_r", "lin_ref"]:
    out[k] = rv.summarize(q, np.array(res[k]), NS)
d_full = np.array(res["anova"]) - q
d_q = np.array(res["anova_q"]) - q
ext = (4 * d_full - d_q) / 3                                       # bias ~ a + b/NI  ->  NI = infinity
out["anova_extrap_inf"] = dict(bias_pp=float(100 * ext.mean()), se_pp=float(100 * ext.std(ddof=1) / np.sqrt(NS)),
                               abs_pp=float(100 * np.abs(ext).mean()))
out["secs"] = round(time.time() - t0)
print(json.dumps(out, indent=1))
np.savez(f"{rv.OUT}/anova_{name}_K{K}.npz", **{k: np.array(v) for k, v in res.items()})
json.dump(out, open(f"{__file__.rsplit('/', 1)[0]}/anova_{name}_K{K}.json", "w"), indent=1)
