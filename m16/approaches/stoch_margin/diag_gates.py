"""Diagnostic 2 (random inputs, DIAGNOSIS ONLY): could a small mixture over gate patterns carry the low inputs?

For each top-3 skeleton S, draw M sets of low inputs and record the full gate pattern (10 steps x 16 neurons).
Reports, per skeleton: number of distinct patterns, mass of the top-1/5/20 patterns, and an ORACLE mixture
ceiling: draws in the top-k patterns of S are computed exactly (inside its own pattern the net is exactly
linear, so a mixture with exact region masses and exact in-region laws would be exact there); all other draws use
the base-point linearisation ('lin' of diag_cond). Also the share of the conditional error variance explained by
x4 (largest low input) alone, and the skewness of the dominant margin.
usage: python m16/approaches/stoch_margin/diag_gates.py MODEL [n_skel] [M]
"""
import json
import sys

import numpy as np
from scipy.stats import norm, skew

sys.path.insert(0, "m16/approaches/stoch_margin")
import sm  # noqa: E402

name = sys.argv[1]
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
M = int(sys.argv[3]) if len(sys.argv) > 3 else 256
model = sm.get_models([name])[name]
Wi, Wh, Wo = model
rng = np.random.default_rng(11)
CH = 200
KS = (1, 5, 20, 50)
stat = {f"top{k}_mass": [] for k in KS}
stat.update(ndist=[], q=[], v2=[], **{f"oracle{k}": [] for k in KS}, lin=[], x4share_num=[], x4share_den=[], skew=[])


def run_gates(x):
    h = np.zeros((len(x), 16)); G = np.empty((len(x), 10, 16), bool)
    for t in range(10):
        pre = h @ Wh.T + x[:, t:t + 1] * Wi[None]
        G[:, t] = pre > 0
        h = np.maximum(pre, 0)
    return h @ Wo.T, G


for c in range(NS // CH):
    pos, val = sm.sample_skeletons(CH, rng)
    v3 = val[:, 2]
    low = sm.trunc_below(v3[:, None, None], rng.uniform(size=(CH, M, 7)))
    x, free = sm.build_inputs(pos, val, low)
    p2 = pos[:, 1]
    L, G = run_gates(x.reshape(-1, 10))
    L = L.reshape(CH, M, 10)
    keys = np.packbits(G.reshape(CH, M, 160), axis=2)
    keys = keys.view(np.dtype((np.void, keys.shape[2]))).reshape(CH, M)
    ok = L.argmax(2) == p2[:, None]
    mu = -norm.pdf(v3) / norm.cdf(v3)
    xb = x[:, 0].copy()
    for j in range(7):
        xb[np.arange(CH), free[:, j]] = mu
    L0 = sm.logits(model, xb)
    J = np.empty((CH, 10, 7))
    for j in range(7):
        xe = xb.copy(); xe[np.arange(CH), free[:, j]] += 1e-6
        J[:, :, j] = (sm.logits(model, xe) - L0) / 1e-6
    Llin = L0[:, None] + np.einsum("nkj,nmj->nmk", J, low - mu[:, None, None])
    oklin = Llin.argmax(2) == p2[:, None]
    for i in range(CH):
        u, inv, cnt = np.unique(keys[i], return_inverse=True, return_counts=True)
        order = np.argsort(-cnt)
        rank = np.empty(len(u), int); rank[order] = np.arange(len(u))
        r = rank[inv]
        stat["ndist"].append(len(u)); stat["q"].append(ok[i].mean()); stat["v2"].append(val[i, 1])
        stat["lin"].append(oklin[i].mean())
        for k in KS:
            stat[f"top{k}_mass"].append(cnt[order[:k]].sum() / M)
            stat[f"oracle{k}"].append(np.where(r < k, ok[i], oklin[i]).mean())
        err = ~ok[i]
        if 0.02 < err.mean() < 0.98:
            x4 = low[i].max(1)
            b = np.digitize(x4, np.quantile(x4, np.linspace(0, 1, 9)[1:-1]))
            cm = np.array([err[b == k].mean() for k in range(8)]); nb = np.array([(b == k).sum() for k in range(8)])
            stat["x4share_num"].append((nb * (cm - err.mean()) ** 2).sum() / M)
            stat["x4share_den"].append(err.var())
            p3 = pos[i, 2]
            stat["skew"].append(skew(L[i, :, p2[i]] - L[i, :, p3]))
S = {k: np.array(v) for k, v in stat.items()}
q = S["q"]; w = 1 - q  # weight skeletons by error mass for the concentration stats
out = dict(model=name, NS=NS, M=M,
           ndist_median=float(np.median(S["ndist"])), ndist_errweighted=float((S["ndist"] * w).sum() / w.sum()),
           **{f"top{k}_mass_errweighted": float((S[f"top{k}_mass"] * w).sum() / w.sum()) for k in KS},
           lin_bias_pp=float(100 * (S["lin"] - q).mean()), lin_abs_pp=float(100 * np.abs(S["lin"] - q).mean()),
           **{f"oracle{k}_bias_pp": float(100 * (S[f"oracle{k}"] - q).mean()) for k in KS},
           **{f"oracle{k}_abs_pp": float(100 * np.abs(S[f"oracle{k}"] - q).mean()) for k in KS},
           x4_share_of_cond_err_var=float(S["x4share_num"].sum() / S["x4share_den"].sum()),
           margin23_skew_median=float(np.median(S["skew"])))
print(json.dumps(out, indent=1))
json.dump(out, open(f"m16/approaches/stoch_margin/diag_gates_{name}.json", "w"), indent=1)
