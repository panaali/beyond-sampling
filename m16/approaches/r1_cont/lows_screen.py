"""T365 stage 1: oracle screen of spread-lows / v4-position closures on T363's 8 failing models x 4 t (README, T365
pre-registration). Same sampler as e2_oracle; float32 forwards, 250k conditional samples per case.
-> results_lows_screen.json"""
import json
import os
import sys
import time

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, ndtri

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cont_e as E  # noqa: E402
import e2_oracle as O  # noqa: E402

N = 250_000
PAR = np.array([(-1.0) ** p for p in range(10)])
# model-free tables: sd of N | < v4, and the expected order statistics of 6 iid N | < v4 (descending), on a v4 grid
V4G = np.linspace(-9, 5, 1401)
sig = lambda v: np.sqrt(np.maximum(1 - v * O.mu(v) - O.mu(v) ** 2 * 0 - O.mu(v) ** 2 + 0.0, 0) if False else None)
def _sd(v):
    m = O.mu(v); return np.sqrt(np.maximum(1 + v * m - m * m, 0.0))   # Var(X | X < v) = 1 + v mu - mu^2, mu = -phi/Phi
u = np.linspace(0, 1, 4001)[1:-1]
from scipy.stats import beta  # noqa: E402  (model-free table only)
_os = np.zeros((len(V4G), 6))
for k in range(6):                                                    # k = 0: largest of 6
    f = beta.pdf(u, 6 - k, k + 1); f /= f.sum()
    _os[:, k] = (ndtri(ndtr(V4G)[:, None] * u[None]) * f[None]).sum(1)
ARR = [np.arange(6), np.arange(6)[::-1], np.array([0, 5, 1, 4, 2, 3]), np.array([5, 0, 4, 1, 3, 2])]


def forward_wrong(model32, x, tri):
    Wi, Wh, Wo = model32; h = np.zeros((len(x), 16), np.float32); x = x.astype(np.float32)
    for s in range(10):
        h = np.maximum(h @ Wh.T + x[:, s:s + 1] * Wi, 0)
    return ((h @ Wo.T).argmax(1) != E.TRI[tri, 1]).astype(np.float64)


def arm_wrong(arm, model32, v1, v2, v3, tri):
    n = len(v2); r = np.arange(n); T3 = E.TRI[tri]
    v3s = np.interp(v2, O.V2G, O._v3s); v4s = np.interp(v2, O.V2G, O._v4s); v4 = v3 - (v3s - v4s)
    free = np.concatenate([E.P4[tri][:, None], E.LOWS[tri]], 1)                 # latest free position first
    lows_nodes = []
    if arm in ("A1", "A3"):
        m, s = O.mu(v4), _sd(v4)
        for sgn in (1.0, -1.0):
            lows_nodes.append(lambda pos, sgn=sgn: m[:, None] + sgn * PAR[pos] * s[:, None])
    else:
        os_ = np.stack([np.interp(v4, V4G, _os[:, k]) for k in range(6)], 1)    # (n, 6) descending
        for a in ARR:
            lows_nodes.append(lambda pos, a=a: os_[:, a])
    v4pos = range(7) if arm in ("A3", "A4") else [0]
    acc = np.zeros(n); cnt = 0
    for f in v4pos:
        p4 = free[:, f]; lowpos = np.delete(free, f, axis=1); lowpos = np.sort(lowpos, 1)
        for node in lows_nodes:
            x = np.empty((n, 10)); x[r, T3[:, 0]] = v1; x[r, T3[:, 1]] = v2; x[r, T3[:, 2]] = v3; x[r, p4] = v4
            x[r[:, None], lowpos] = node(lowpos)
            acc += forward_wrong(model32, x, tri); cnt += 1
    return float((acc / cnt).mean())


if __name__ == "__main__":
    z = np.load(os.path.join(HERE, "..", "..", "..", "data", "dev3", "weights.npz")); names = [str(s) for s in z["names"]]
    truth = json.load(open(os.path.join(HERE, "..", "..", "..", "data", "t363", "t358_rare_mc.json")))
    fn = os.path.join(HERE, "results_lows_screen.json"); out = json.load(open(fn)) if os.path.exists(fn) else {}
    for nm in O.MODELS:
        i = names.index(nm); m32 = tuple(np.asarray(w, np.float32) for w in (z["Wi"][i], z["Wh"][i], z["Wo"][i]))
        for t in O.TS:
            key = f"{nm}|{t}"
            if key in out:
                continue
            t0 = time.time(); W = quad(lambda v: 90 * O.pdf(v) * ndtr(-(v + t)) * ndtr(v - t) ** 8, -12, 12, epsrel=1e-12, limit=400)[0]
            tag = {0.5: "E2_0p5", 0.75: "E2_0p75", 1.0: "E2_1", 1.25: "E2_1p25"}[t]; p = truth[nm]["counts"][tag] / truth[nm]["n"]
            rng = np.random.default_rng(365000 + i * 10 + int(t * 4)); v1, v2, v3, _, tri = O.sample(t, N, rng)
            rec = {a: W * arm_wrong(a, m32, v1, v2, v3, tri) / p for a in ("A1", "A2", "A3", "A4")}
            out[key] = rec; json.dump(out, open(fn, "w"), indent=1)
            print(key, " ".join(f"{a} {v:.2f}" for a, v in rec.items()), round(time.time() - t0), flush=True)
    for a in ("A1", "A2", "A3", "A4"):
        v = [r[a] for r in out.values()]; print(a, "within 2x", sum(.5 <= x <= 2 for x in v), "/", len(v), "median %.2f" % np.median(v))
