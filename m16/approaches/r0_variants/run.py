"""T298: paired oracle biases of point closures of the low inputs on the 80 DEV2 retrains (diagnostic sampling)."""
import hashlib
import json
import sys
import time
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import norm

N = int(sys.argv[1]) if len(sys.argv) > 1 else 500_000
EXPLORE = len(sys.argv) > 2 and sys.argv[2] == "explore"
z = np.load("/home/user/dl/dev2/weights.npz", allow_pickle=True)
assert hashlib.sha256(z["Wi"].tobytes() + z["Wh"].tobytes() + z["Wo"].tobytes()).hexdigest().startswith("22d1c7c2")
names = [str(n) for n in z["names"]]
T = 10


def predict(model, x):
    Wi, Wh, Wo = model
    h = np.zeros((len(x), Wh.shape[0]))
    for t in range(T):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    return (h @ Wo.T).argmax(1)


def closures(x):
    """dict name -> list of closure point arrays (averaged if > 1)."""
    n = len(x)
    o = np.argsort(-x, 1)
    v = np.take_along_axis(x, o, 1)
    r = np.arange(n)
    out = {}
    mu3 = -norm.pdf(v[:, 2]) / ndtr(v[:, 2])
    mu4 = -norm.pdf(v[:, 3]) / ndtr(v[:, 3])
    med3 = ndtri(ndtr(v[:, 2]) / 2)
    low3 = np.zeros((n, T), bool)
    low3[r[:, None], o[:, 3:]] = True
    low4 = np.zeros((n, T), bool)
    low4[r[:, None], o[:, 4:]] = True
    out["a_R0"] = [np.where(low3, mu3[:, None], x)]
    out["b_top4"] = [np.where(low4, mu4[:, None], x)]
    out["c_median"] = [np.where(low3, med3[:, None], x)]
    # d': canonical spread values below v3, descending or ascending in time over the free positions
    spread = ndtri(ndtr(v[:, 2])[:, None] * (np.arange(7, 0, -1) / 8)[None])      # (n,7) descending values
    free = np.sort(o[:, 3:], 1)                                                     # free positions in time order
    xd, xa = x.copy(), x.copy()
    xd[r[:, None], free] = spread
    xa[r[:, None], free] = spread[:, ::-1]
    out["dprime_spread"] = [xd, xa]
    # e: R0 with the latest free position exact
    xe = np.where(low3, mu3[:, None], x)
    last = free[:, -1]
    xe[r, last] = x[r, last]
    out["e_R0_latest"] = [xe]
    if EXPLORE:                                   # post hoc combinations (not pre-registered)
        free4 = np.sort(o[:, 4:], 1)
        xf = np.where(low4, mu4[:, None], x)
        xf[r, free4[:, -1]] = x[r, free4[:, -1]]
        out["x_top4_latest"] = [xf]
        sp4 = ndtri(ndtr(v[:, 3])[:, None] * (np.arange(6, 0, -1) / 7)[None])
        x4d, x4a = x.copy(), x.copy()
        x4d[r[:, None], free4] = sp4
        x4a[r[:, None], free4] = sp4[:, ::-1]
        out["x_top4_spread"] = [x4d, x4a]
        for cn in ("a_R0", "b_top4", "c_median", "dprime_spread", "e_R0_latest"):
            out.pop(cn)
    target = o[:, 1]
    g12, g23 = v[:, 0] - v[:, 1], v[:, 1] - v[:, 2]
    reg = np.where(v[:, 1] <= 0, 0, np.where(np.minimum(g12, g23) >= .05, 3, np.where(g23 <= g12, 1, 2)))
    return out, target, reg


REG = ("blind", "band23", "band12", "tail")
res = {}
for k, name in enumerate(names):
    model = (z["Wi"][k].astype(float), z["Wh"][k].astype(float), z["Wo"][k].astype(float))
    rng = np.random.default_rng(298000 + k)
    t0 = time.time()
    acc = {}
    sums = {}
    for c in range(0, N, 100_000):
        x = rng.standard_normal((min(100_000, N - c), T))
        cl, tgt, reg = closures(x)
        ok = (predict(model, x) == tgt).astype(float)
        acc["ok"] = acc.get("ok", 0) + ok.sum()
        for cn, pts in cl.items():
            okc = np.mean([predict(model, p) == tgt for p in pts], 0)
            d = okc - ok
            s = sums.setdefault(cn, {"d": 0.0, "d2": 0.0, **{r_: 0.0 for r_ in REG}})
            s["d"] += d.sum(); s["d2"] += (d ** 2).sum()
            for i, r_ in enumerate(REG):
                s[r_] += d[reg == i].sum()
    res[name] = {"acc_sample": acc["ok"] / N}
    for cn, s in sums.items():
        m = s["d"] / N
        res[name][cn] = {"bias_pp": 100 * m, "se_pp": 100 * np.sqrt((s["d2"] / N - m * m) / N),
                         **{f"{r_}_pp": 100 * s[r_] / N for r_ in REG}}
    print(k, name, " ".join(f"{cn}:{res[name][cn]['bias_pp']:+.3f}" for cn in sums), "%.0fs" % (time.time() - t0), flush=True)
    json.dump(res, open("results/biases_explore.json" if EXPLORE else "results/biases.json", "w"), indent=1)
