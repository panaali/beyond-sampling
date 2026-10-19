"""T427 checks (SAMPLING, validation only): T401's conditional-law MC (rare_events/truthc.py + rating_mc.py) generalised
to T inputs. x is drawn exactly from its law given g23 >= t: v2 ~ T(T-1) phi (1 - Phi) Phi(v2 - t)^(T-2), v1 | v2 ~ N > v2,
v3 = Phi^-1(Phi(v2 - t) U^(1/(T-2))), T-3 lows iid N | < v3, uniform random positions. Hit = pick == pos(v3); sampled
tie share = hits with g12 < .1. Adaptive: stop at >= HMAX hits or NMAX samples.
usage: python mc.py TAG [TAG ...] -> results/mc_<TAG>.json (one file per model; resumable per grid point)"""
import json
import os
import sys
import time

import numpy as np
from scipy.special import ndtr, ndtri

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import grid  # noqa: E402
import t421  # noqa: E402

GRID = np.linspace(-7.0, 10.0, 2_000_001)
NMAX, HMAX, CHUNK = 100_000_000, 1000, 250_000


def sampler(t0, T):
    f = T * (T - 1) * np.exp(-GRID ** 2 / 2) / np.sqrt(2 * np.pi) * ndtr(-GRID) * ndtr(GRID - t0) ** (T - 2)
    c = np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) / 2 * np.diff(GRID))]); P = c[-1]; c /= P

    def draw(rng, n):
        v2 = np.interp(rng.random(n), c, GRID)
        v1 = -ndtri(rng.random(n) * ndtr(-v2))
        v3 = ndtri(ndtr(v2 - t0) * rng.random(n) ** (1 / (T - 2)))
        lows = ndtri(ndtr(v3)[:, None] * rng.random((n, T - 3)))
        vals = np.column_stack([v1, v2, v3, lows]); perm = np.argsort(rng.random((n, T)), 1)
        x = np.empty_like(vals); x[np.arange(n)[:, None], perm] = vals
        return x, perm[:, 2], v1 - v2
    return draw, float(P)


def pick(m, x):
    Wi, Wh, Wo = m; WhT = Wh.T.copy(); h = np.zeros((len(x), len(Wi)))
    for s in range(x.shape[1]):
        h = h @ WhT; h += x[:, s:s + 1] * Wi; np.maximum(h, 0, out=h)
    return (h @ Wo.T).argmax(1)


def run(m, T, t, seed):
    draw, P = sampler(t, T); rng = np.random.default_rng(seed); n = hits = tie = 0
    while n < NMAX and hits < HMAX:
        x, p3, g12 = draw(rng, CHUNK); h = pick(m, x) == p3
        hits += int(h.sum()); tie += int((h & (g12 < 0.1)).sum()); n += CHUNK
    q = hits / n
    return dict(t=t, n=n, hits=hits, P_cond=P, rate=q, rate_se=float(np.sqrt(max(q * (1 - q), 1e-300) / n)),
                p_joint=P * q, tie_share=(tie / hits if hits else None))


if __name__ == "__main__":
    for tag in sys.argv[1:]:
        m, rec, h = t421.load(tag); T = rec["T"]; path = os.path.join(HERE, "results", f"mc_{tag}.json")
        out = json.load(open(path)) if os.path.exists(path) else dict(tag=tag, sha256=h, pts={})
        for i, t in enumerate(grid.tgrid(T)):
            if f"{t:.6f}" in out["pts"]:
                continue
            t0 = time.time(); r = run(m, T, t, seed=427_000 + 1000 * i + sum(map(ord, tag))); r["sec"] = time.time() - t0
            out["pts"][f"{t:.6f}"] = r; json.dump(out, open(path, "w"), indent=1)
            print(tag, i, f"t {t:.3f} n {r['n']:.2e} hits {r['hits']} rate {r['rate']:.3e} tie {r['tie_share']} {r['sec']:.0f}s", flush=True)
