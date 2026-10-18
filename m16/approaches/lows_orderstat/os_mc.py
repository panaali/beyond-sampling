"""T341: oracle closures of the lows by order statistics vs T327's spread; T298 protocol on 80 DEV2. Diagnostic."""
import json
import sys
import numpy as np
from scipy.special import ndtr

phi = lambda v: np.exp(-v * v / 2) / np.sqrt(2 * np.pi)
mu = lambda u: -phi(u) / np.maximum(ndtr(u), 1e-300)
sd = lambda u: np.sqrt(np.maximum(1 - u * phi(u) / np.maximum(ndtr(u), 1e-300) - mu(u) ** 2, 0))
GRID = np.linspace(-4.5, 4.0, 851)
_u = [np.linspace(g - 10, g, 4001) for g in GRID]
RK = np.array([[g - np.trapezoid((ndtr(u) / ndtr(g)) ** k, u) for g, u in zip(GRID, _u)] for k in range(1, 7)])
rk = lambda k, v4: np.interp(v4, GRID, RK[k - 1])                          # E[max of k iid N | < v4]


def ok(model, X, k):
    Wi, Wh, Wo = model
    h = np.zeros((len(X), Wh.shape[0]))
    for s in range(10):
        h = np.maximum(h @ Wh.T + X[:, s:s + 1] * Wi[None], 0)
    return (h @ Wo.T).argmax(1) == k


def closures(x):
    n = len(x); r = np.arange(n)[:, None]
    P = np.argsort(-x, 1); top = P[:, :3]
    isF = np.ones((n, 10), bool); isF[r, top] = False
    F = np.sort(np.where(isF, np.arange(10)[None], 99), 1)[:, :7]           # free positions, time order
    xf = x[r, F]                                                           # free values in time order
    srt = -np.sort(-xf, 1); v4, v5 = srt[:, 0], srt[:, 1]
    a4 = np.argmax(xf, 1); xf2 = xf.copy(); xf2[np.arange(n), a4] = -np.inf; a5 = np.argmax(xf2, 1)
    out = {}
    def put(vals):
        X = x.copy(); X[r, F] = vals; return X
    # (c) closure c
    v = np.repeat(mu(v4)[:, None], 7, 1); v[:, 6] = v4; out["c"] = put(v)
    # (i) v5 at true position, v4 latest (or next-latest if v5 is latest)
    v = np.repeat(mu(v5)[:, None], 7, 1); last = np.where(a5 == 6, 5, 6)
    v[np.arange(n), a5] = v5; v[np.arange(n), last] = v4; out["i"] = put(v)
    # (ii) v5 second-latest, v4 latest
    v = np.repeat(mu(v5)[:, None], 7, 1); v[:, 5] = v5; v[:, 6] = v4; out["ii"] = put(v)
    # (iii) true running max: records keep value, non-records at mu(running max so far)
    rm = np.maximum.accumulate(xf, 1); rec = xf >= rm
    prev = np.concatenate([np.full((n, 1), -np.inf), rm[:, :-1]], 1)
    v = np.where(rec, xf, mu(np.where(np.isfinite(prev), prev, 0.0))); out["iii"] = put(v)
    # (iv) running-max profile rule
    v = np.stack([rk(k, v4) for k in range(1, 7)] + [v4], 1); out["iv"] = put(v)
    # (s+), (s-) spread on the 6 non-v4 lows, absolute position parity
    sg = (-1.0) ** F[:, :6]
    for nm, sgn in (("s+", 1), ("s-", -1)):
        v = np.empty((n, 7)); v[:, :6] = mu(v4)[:, None] + sgn * sg * sd(v4)[:, None]; v[:, 6] = v4
        out[nm] = put(v)
    out["_lastpar"] = F[:, 6] % 2
    return out


if __name__ == "__main__":
    Z = np.load("/home/user/dl/dev2/weights.npz", allow_pickle=True)
    truth = {k: v["acc"] for k, v in json.load(open("/home/user/dl/dev2/truths.json"))["truths"].items()}
    idx = range(len(Z["names"])) if len(sys.argv) < 2 else [int(a) for a in sys.argv[1:]]
    N, CH = 500_000, 50_000; V = ["c", "i", "ii", "iii", "iv", "s+", "s-"]; res = {}
    for i in idx:
        name = str(Z["names"][i]); m = tuple(np.asarray(Z[w][i], float) for w in ("Wi", "Wh", "Wo"))
        rng = np.random.default_rng(341000 + i); S = {v: 0.0 for v in V}; S2 = dict(S)
        par = {f"{v}_par{p}": 0.0 for v in ("s+", "s-") for p in (0, 1)}; npar = [0, 0]
        for _ in range(N // CH):
            x = rng.standard_normal((CH, 10)); k = np.argsort(-x, 1)[:, 1]; o = ok(m, x, k)
            C = closures(x); lp = C.pop("_lastpar")
            for p in (0, 1):
                npar[p] += int((lp == p).sum())
            for v in V:
                d = ok(m, C[v], k).astype(float) - o; S[v] += d.sum(); S2[v] += (d * d).sum()
                if v in ("s+", "s-"):
                    for p in (0, 1):
                        par[f"{v}_par{p}"] += d[lp == p].sum()
        b = {v: S[v] / N for v in V}
        res[name] = {"truth": truth[name], "b": b, "se": {v: float(np.sqrt((S2[v] / N - b[v] ** 2) / N)) for v in V},
                     "spread_by_lastparity": {kk: vv / npar[int(kk[-1])] for kk, vv in par.items()}}
        print(name, {v: round(b[v] * 100, 3) for v in V}, flush=True)
    json.dump(res, open(f"results/os_{'all' if len(idx) > 1 else idx[0]}.json", "w"), indent=1)
