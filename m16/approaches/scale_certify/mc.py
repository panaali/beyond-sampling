"""T426 step 1: plain MC of E1(t0) inside the far-gap slab S = {g23 >= t0} (g23 = 2nd largest - 3rd largest x).
Exact conditional sampler (model-free): every triple has the same slab mass, so draw the triple uniformly; given the
triple, x_p3 = v has density prop. to phi(v) Phi(v)^(T-3) Phibar(v+t0)^2 (sampled by inverse CDF on a 4e5-point grid),
the T-3 lows are iid N(0,1) truncated to <= v, the two highs iid N(0,1) truncated to >= v + t0 (larger -> p1).
P(E1) = P(S) * hits / n; P(S) = T(T-1)(T-2)/2 * int phi Phi^(T-3) Phibar(v+t0)^2 dv.
usage: python mc.py TAG T0 N [SEED] -> results/mc_<TAG>_<t0>_<seed>.json (summary); data/t426/mc/ (same plus hit x)"""
import json
import os
import sys
import time
from collections import Counter

import numpy as np
from scipy.special import log_ndtr, ndtri
from scipy.stats import beta, norm

import core

HERE = os.path.dirname(os.path.abspath(__file__))


def vgrid(T, t0):
    v = np.linspace(-9, 7, 400_001)
    lf = norm.logpdf(v) + (T - 3) * log_ndtr(v) + 2 * log_ndtr(-(v + t0))
    f = np.exp(lf); dv = v[1] - v[0]
    c = np.concatenate([[0], np.cumsum((f[1:] + f[:-1]) / 2 * dv)])
    return v, c, c[-1]


def slab_mass(T, t0):
    return T * (T - 1) * (T - 2) / 2 * vgrid(T, t0)[2]


def t0_rule(T, q=None):
    """Model-free threshold: the t0 whose slab mass equals T = 10's slab mass at t0 = 2.5."""
    q = q or slab_mass(10, 2.5)
    lo, hi = 0.5, 6.0
    for _ in range(60):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if slab_mass(T, m) > q else (lo, m)
    return round((lo + hi) / 2, 4)


def draw(rng, n, T, t0, grid):
    v, c, Z = grid
    keys = rng.random((n, T)).argsort(1); tri = keys[:, :3]           # uniform ordered triple; rest = lows
    x3 = np.interp(rng.random(n) * Z, c, v)
    lows = ndtri(rng.random((n, T - 3)) * np.exp(log_ndtr(x3))[:, None])
    a = -(x3 + t0)                                                     # highs: -ndtri(U Phi(a))
    hi = -ndtri(rng.random((n, 2)) * np.exp(log_ndtr(a))[:, None])
    hi.sort(1)
    x = np.empty((n, T)); r = np.arange(n)[:, None]
    x[r, keys[:, 3:]] = lows; x[r[:, 0], tri[:, 2]] = x3; x[r[:, 0], tri[:, 1]] = hi[:, 0]; x[r[:, 0], tri[:, 0]] = hi[:, 1]
    return x, tri


def main():
    tag, t0, n = sys.argv[1], float(sys.argv[2]), int(float(sys.argv[3])); seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    core.set_model(tag); T = core.D; grid = vgrid(T, t0); ps = slab_mass(T, t0)
    rng = np.random.default_rng(426_000 + seed); C = Counter(); hits = []; done = 0; t1 = time.time(); chunk = 200_000
    while done < n:
        m = min(chunk, n - done); x, tri = draw(rng, m, T, t0, grid)
        assert (np.sort(x, 1)[:, -2] - np.sort(x, 1)[:, -3] >= t0 - 1e-9).all()
        L = core.run(x)[:, -1] @ core.Wo.T; ev = L.argmax(1) == tri[:, 2]
        for i in np.flatnonzero(ev):
            C[tuple(int(u) for u in tri[i])] += 1; hits.append(x[i].tolist())
        done += m
    k = sum(C.values())
    out = dict(tag=tag, H=core.H, T=T, t0=t0, seed=seed, n=n, slab_mass=ps, hits=k,
               p=ps * k / n, se=ps * np.sqrt(k) / n,
               cp95=ps * float(beta.ppf(.95, k + 1, n - k)), cp1e6=ps * float(beta.ppf(1 - 1e-6, k + 1, n - k)),
               by_triple=sorted(([list(t), c] for t, c in C.items()), key=lambda r: -r[1]), sec=round(time.time() - t1, 1),
               hit_x=hits)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "..", "..", "..", "data", "t426", "mc", f"mc_{tag}_{t0:g}_{seed}.json"), "w"))
    json.dump({k_: v_ for k_, v_ in out.items() if k_ != "hit_x"}, open(os.path.join(HERE, "results", f"mc_{tag}_{t0:g}_{seed}.json"), "w"))
    print({k_: v_ for k_, v_ in out.items() if k_ != "hit_x"} | {"by_triple": out["by_triple"][:8]})


if __name__ == "__main__":
    main()
