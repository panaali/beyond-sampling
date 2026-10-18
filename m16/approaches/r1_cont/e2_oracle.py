"""T363: oracle MC chain for E2(t) = {wrong, g12 >= t, g23 >= t} (README, T363 pre-registration). Sampling oracle only.
usage: python e2_oracle.py -> results_e2_oracle.json"""
import json
import os
import sys
import time

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, ndtri

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cont_e as E  # noqa: E402  (TRI, P4, LOWS, gauss_v4, mu; model-free)

MODELS = ["dev3_s508", "dev3_s511", "dev3_s512", "dev3_s515", "dev3_s517", "dev3_s518", "dev3_s519", "dev3_s525"]
TS = (0.5, 0.75, 1.0, 1.25)
N = 1_000_000
CH = 250_000
pdf = lambda v: np.exp(-v * v / 2) / np.sqrt(2 * np.pi)


def cond_g23(v2, n=20001):                              # E[v2 - v3 | v2] (as T340's _cond_g23)
    v = np.linspace(v2 - 12, v2, n); f = (ndtr(v) / ndtr(v2)) ** 8
    return float(((f[1:] + f[:-1]) / 2 * np.diff(v)).sum())


# model-free closure tables on grids
V2G = np.linspace(-6, 6, 1201)
_v3s = np.array([v - cond_g23(v) for v in V2G]); _v4s = np.array([float(E.gauss_v4(v, 1)[0][0]) for v in _v3s])
V3G = np.linspace(-9, 6, 1501); _ev4 = np.array([float(E.gauss_v4(v, 1)[0][0]) for v in V3G])
mu = lambda v: -pdf(v) / np.maximum(ndtr(v), 1e-300)
dmu = lambda v: -v * mu(v) + mu(v) ** 2


def sample(t, n, rng):
    g = np.linspace(-8, 8, 160001); w = 90 * pdf(g) * ndtr(-(g + t)) * ndtr(g - t) ** 8
    c = np.cumsum(w); c /= c[-1]; v2 = np.interp(rng.random(n), c, g)
    v1 = -ndtri(rng.random(n) * ndtr(-(v2 + t)))
    v3 = ndtri(ndtr(v2 - t) * rng.random(n) ** (1 / 8))
    lows7 = ndtri(ndtr(v3)[:, None] * rng.random((n, 7)))           # 7 iid N below v3
    tri = rng.integers(len(E.TRI), size=n)
    return v1, v2, v3, lows7, tri


def build(world, v1, v2, v3, lows7, tri, rng):
    n = len(v2); x = np.empty((n, 10)); r = np.arange(n); T3 = E.TRI[tri]
    x[r, T3[:, 0]] = v1; x[r, T3[:, 1]] = v2; x[r, T3[:, 2]] = v3
    P4 = E.P4[tri]; LOW = E.LOWS[tri]
    if world == "W0":
        v3s = np.interp(v2, V2G, _v3s); v4s = np.interp(v2, V2G, _v4s); v4 = v3 - (v3s - v4s)
        lo = mu(v4s) + dmu(v4s) * (v4 - v4s); lows = np.repeat(lo[:, None], 6, 1)
    elif world == "W1":
        v4 = np.interp(v3, V3G, _ev4); lows = np.repeat(mu(v4)[:, None], 6, 1)
    elif world == "W2":
        v4 = lows7.max(1); lows = np.repeat(mu(v4)[:, None], 6, 1)
    elif world == "W3":
        v4 = lows7.max(1); srt = np.sort(lows7, 1); lows = srt[:, :6]          # the other 6 are iid below v4 given v4
        lows = lows[r[:, None], np.argsort(rng.random((n, 6)), 1)]
    else:                                                                  # W4: random positions for all 7
        free = np.concatenate([P4[:, None], LOW], 1); perm = np.argsort(rng.random((n, 7)), 1)
        x[r[:, None], free[r[:, None], perm]] = lows7
        return x
    x[r, P4] = v4; x[r[:, None], LOW] = lows
    return x


def wrong_frac(model, x, tri):
    Wi, Wh, Wo = model; h = np.zeros((len(x), 16))
    for s in range(10):
        h = np.maximum(h @ Wh.T + np.outer(x[:, s], Wi), 0)
    return float(((h @ Wo.T).argmax(1) != E.TRI[tri, 1]).mean())


if __name__ == "__main__":
    z = np.load(os.path.join(HERE, "..", "..", "..", "data", "dev3", "weights.npz")); names = [str(s) for s in z["names"]]
    truth = json.load(open(os.path.join(HERE, "..", "..", "..", "data", "t363", "t358_rare_mc.json")))
    t360 = json.load(open(os.path.join(HERE, "..", "..", "..", "data", "t363", "t360_score.json")))["K3"]["cases"]
    fn = os.path.join(HERE, "results_e2_oracle.json"); out = json.load(open(fn)) if os.path.exists(fn) else {}
    for nm in MODELS:
        i = names.index(nm); model = (z["Wi"][i].astype(float), z["Wh"][i].astype(float), z["Wo"][i].astype(float))
        for t in TS:
            key = f"{nm}|{t}"
            if key in out:
                continue
            t0 = time.time(); W = quad(lambda v: 90 * pdf(v) * ndtr(-(v + t)) * ndtr(v - t) ** 8, -12, 12, epsrel=1e-12, limit=400)[0]
            tag = {0.5: "E2_0p5", 0.75: "E2_0p75", 1.0: "E2_1", 1.25: "E2_1p25"}[t]
            p_true = truth[nm]["counts"][tag] / truth[nm]["n"]; rec = dict(weight=W, truth=p_true, k3_t360=t360.get(f"{nm}|{tag}", {}).get("est"))
            acc = {w: 0.0 for w in ("W0", "W1", "W2", "W3", "W4")}
            rng = np.random.default_rng(363000 + i * 10 + int(t * 4))
            for s in range(0, N, CH):
                v1, v2, v3, lows7, tri = sample(t, CH, rng)
                for w in acc:
                    acc[w] += wrong_frac(model, build(w, v1, v2, v3, lows7, tri, rng), tri) * CH / N
            for w, q in acc.items():
                rec[w] = dict(p=W * q, ratio=W * q / p_true, se_rel=float(np.sqrt((1 - q) / (q * N))) if q > 0 else None)
            rec["k3_ratio"] = rec["k3_t360"] / p_true if rec["k3_t360"] else None
            out[key] = rec; json.dump(out, open(fn, "w"), indent=1)
            print(key, "K3 %.2f" % (rec["k3_ratio"] or -1), " ".join(f"{w} {rec[w]['ratio']:.2f}" for w in acc), round(time.time() - t0), flush=True)
