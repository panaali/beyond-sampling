"""T401 validation (SAMPLING, diagnostic): conditional-law MC per grid point, adaptive (stop at >= 1000 hits or 1e8 samples).
E1: x | g23 >= t (truthc sampler), hit = pick == p3, triple-tie share = hits with g12 < .1.
E2: x | g12 >= t, g23 >= t (v2 ~ 90 phi (1 - Phi(v2 + t)) Phi(v2 - t)^8, v1 > v2 + t, v3 < v2 - t, lows | < v3),
hit = pick != p2. -> results/t401_mc.json"""
import json
import os
import sys
import time

import numpy as np
from scipy.special import ndtr, ndtri

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from deep import models  # noqa: E402
from truthc import GRID, sampler, p_gap  # noqa: E402


def e2_sampler(t):
    f = 90 * np.exp(-GRID ** 2 / 2) / np.sqrt(2 * np.pi) * ndtr(-(GRID + t)) * ndtr(GRID - t) ** 8
    c = np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) / 2 * np.diff(GRID))]); P = c[-1]; c /= P

    def draw(rng, n):
        v2 = np.interp(rng.random(n), c, GRID)
        v1 = -ndtri(rng.random(n) * ndtr(-(v2 + t)))
        v3 = ndtri(ndtr(v2 - t) * rng.random(n) ** (1 / 8))
        lows = ndtri(ndtr(v3)[:, None] * rng.random((n, 7)))
        vals = np.column_stack([v1, v2, v3, lows]); perm = np.argsort(rng.random((n, 10)), 1)
        x = np.empty_like(vals); x[np.arange(n)[:, None], perm] = vals
        return x, perm
    return draw, P


def pick(m, x):
    Wi, Wh, Wo = m; h = np.zeros((len(x), 16))
    for s in range(10):
        h = h @ Wh.T; h += x[:, s:s + 1] * Wi; np.maximum(h, 0, out=h)
    return (h @ Wo.T).argmax(1)


def run(m, kind, t, seed, nmax=100_000_000, hmin=1000, chunk=500_000):
    rng = np.random.default_rng(seed); n = hits = tie = 0
    if kind == "E1":
        d = sampler(t); P = p_gap(t)
    else:
        d, P = e2_sampler(t)
    while n < nmax and hits < hmin:
        x, aux = d(rng, chunk); y = pick(m, x)
        if kind == "E1":
            h = y == aux; r = np.arange(chunk); v = -np.sort(-x, 1); tie += int((h & (v[:, 0] - v[:, 1] < 0.1)).sum())
        else:
            h = y != aux[:, 1]
        hits += int(h.sum()); n += chunk
    q = hits / n
    return dict(n=n, hits=hits, P_cond=P, rate=q, rate_se=float(np.sqrt(max(q * (1 - q), 1e-300) / n)),
                p_joint=P * q, tie_share=(tie / hits if (kind == "E1" and hits) else None))


ms = models(); out = {}
for nm in ("orig", "dev3_s529", "dev3_s531"):
    m = tuple(np.asarray(w, float) for w in ms[nm])
    for kind, grid in (("E1", (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)), ("E2", (0.5, 0.75, 1.0, 1.25, 1.5))):
        for i, t in enumerate(grid):
            t0 = time.time(); r = run(m, kind, t, seed=401_000 + 100 * i + (0 if kind == "E1" else 50) + sum(map(ord, nm)))
            r["sec"] = time.time() - t0; out[f"{nm}:{kind}:{t}"] = r
            print(nm, kind, t, {k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
            json.dump(out, open(os.path.join(HERE, "results", "t401_mc.json"), "w"), indent=1)
