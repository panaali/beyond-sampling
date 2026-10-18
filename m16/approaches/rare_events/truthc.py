"""T364 truth: conditional-law MC for E1(t0) = {pick == p3, g23 > t0} (SAMPLING, truth only; independent of the walks).
x is drawn exactly from its law given g23 > t0:
  v2 ~ 90 phi(v2)(1 - Phi(v2)) Phi(v2 - t0)^8 (grid inverse CDF), v1 | v2 ~ N truncated > v2,
  v3 = Phi^-1(Phi(v2 - t0) U^(1/8)), 7 lows iid N | < v3, uniform random positions.
p = P(g23 > t0) x mean(pick == pos(v3)). usage: python truthc.py model:t0:N[,model:t0:N...] -> results/truthc.json"""
import json
import os
import sys
import time

import numpy as np
from scipy.special import ndtr, ndtri

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from deep import models  # noqa: E402

GRID = np.linspace(-7.0, 10.0, 2_000_001)


def dens(t0):
    return 90 * np.exp(-GRID ** 2 / 2) / np.sqrt(2 * np.pi) * ndtr(-GRID) * ndtr(GRID - t0) ** 8


def p_gap(t0):
    f = dens(t0); return float(((f[1:] + f[:-1]) / 2 * np.diff(GRID)).sum())


def sampler(t0):
    f = dens(t0); c = np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) / 2 * np.diff(GRID))]); c /= c[-1]
    def draw(rng, n):
        v2 = np.interp(rng.random(n), c, GRID)
        v1 = -ndtri(rng.random(n) * ndtr(-v2))
        v3 = ndtri(ndtr(v2 - t0) * rng.random(n) ** (1 / 8))
        lows = ndtri(ndtr(v3)[:, None] * rng.random((n, 7)))
        vals = np.column_stack([v1, v2, v3, lows])                    # rank order
        perm = np.argsort(rng.random((n, 10)), 1)                      # perm[:, r] = position of rank r
        x = np.empty_like(vals); x[np.arange(n)[:, None], perm] = vals
        return x, perm[:, 2]
    return draw


def run(model, t0, N, seed, chunk=500_000):
    Wi, Wh, Wo = (np.asarray(w, float) for w in model); WhT, WoT = Wh.T.copy(), Wo.T.copy()
    draw = sampler(t0); rng = np.random.default_rng(seed); hits = 0; n = 0
    while n < N:
        m = min(chunk, N - n); x, p3 = draw(rng, m); h = np.zeros((m, 16))
        for t in range(10):
            h = h @ WhT; h += x[:, t:t + 1] * Wi; np.maximum(h, 0, out=h)
        hits += int(((h @ WoT).argmax(1) == p3).sum()); n += m
    pg = p_gap(t0); q = hits / n
    return dict(t0=t0, n=n, hits=hits, p_gap=pg, p=pg * q, se=pg * np.sqrt(max(q * (1 - q), 1e-300) / n))


def main():
    path = os.path.join(HERE, "results", "truthc.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    ms = models()
    for spec in sys.argv[1].split(","):
        nm, t0, N = spec.split(":"); t0 = float(t0); N = int(float(N)); key = f"{nm}:{t0}"
        t = time.time(); r = run(ms[nm], t0, N, seed=364_000 + int(100 * t0) + sum(map(ord, nm)))
        r["sec"] = time.time() - t
        prev = out.get(key)
        if prev and prev.get("seed_note") is None and prev["n"] and key in out:     # keep separate runs as a list
            out.setdefault(key + ":runs", []).append(prev)
        out[key] = r; json.dump(out, open(path, "w"), indent=1)
        print(key, {k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()}, flush=True)


if __name__ == "__main__":
    main()
