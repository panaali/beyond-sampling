"""T404 truth for E2(t) = {network wrong, g12 >= t, g23 >= t}: own conditional-law MC (SAMPLING, truth only).
Condition on v2, the 2nd largest of 10 iid N(0,1): v2 ~ 90 phi(v) (1 - Phi(v + t)) Phi(v - t)^8 (grid inverse CDF),
v1 ~ N | > v2 + t, the 8 others iid N | < v2 - t, uniform random positions. P(cond) by quadrature; p = P(cond) x
mean(pick != pos(v2)). Same draws through all 52 models (common random numbers). Diagnostic: hits by the rank picked.
usage: python t404_truth.py T N -> results/t404_truth_T.json (checkpointed; resumes)"""
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.special import log_ndtr, ndtr, ndtri

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import models as MP  # noqa: E402
from truth import BLK, pick  # noqa: E402

GRID = np.linspace(-9.0, 7.0, 3_200_001)


def logf(t):
    return np.log(90.0) - GRID ** 2 / 2 - 0.5 * np.log(2 * np.pi) + log_ndtr(-(GRID + t)) + 8 * log_ndtr(GRID - t)


def p_cond(t):
    f = np.exp(logf(t)); return float(((f[1:] + f[:-1]) / 2 * np.diff(GRID)).sum())


def make_draw(t):
    f = np.exp(logf(t)); c = np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) / 2 * np.diff(GRID))]); c /= c[-1]

    def draw(rng, n):
        v2 = np.interp(rng.random(n), c, GRID)
        v1 = -ndtri(rng.random(n) * ndtr(-(v2 + t)))
        lows = ndtri(rng.random((n, 8)) * ndtr(v2 - t)[:, None])
        vals = np.column_stack([v1, v2, lows])                                  # column 1 = v2
        pos = rng.permuted(np.repeat(np.arange(10)[None], n, 0), axis=1)
        x = np.empty_like(vals); x[np.arange(n)[:, None], pos] = vals
        return x
    return draw


def main():
    t = float(sys.argv[1]); N = int(float(sys.argv[2])); chunk = 250_000
    P = MP.panel(); names = sorted(P); torch.set_num_threads(1)
    Ms = [tuple(torch.from_numpy(np.ascontiguousarray(w)) for w in (P[m][0][:, None], P[m][1], P[m][2])) for m in names]
    fn = os.path.join(HERE, "results", f"t404_truth_{t:g}.json")
    st = json.load(open(fn)) if os.path.exists(fn) else dict(t=t, p_cond=p_cond(t), n=0, chunks=0, seed=404_000 + int(round(100 * t)),
                                                             hits={m: 0 for m in names}, by_rank={m: [0] * 10 for m in names}, sec=0.0)
    draw = make_draw(t)
    while st["n"] < N:
        t1 = time.time(); m = min(chunk, N - st["n"]); rng = np.random.default_rng([st["seed"], st["chunks"]])
        x = draw(rng, m); rank = np.argsort(np.argsort(-x, 1), 1); xT = torch.from_numpy(np.ascontiguousarray(x.T))
        for nm, mod in zip(names, Ms):
            pk = np.concatenate([pick(mod, xT[:, a:a + BLK]) for a in range(0, m, BLK)]); rk = rank[np.arange(m), pk]
            st["hits"][nm] += int((rk != 1).sum())
            bc = np.bincount(rk, minlength=10); st["by_rank"][nm] = [a + int(b) for a, b in zip(st["by_rank"][nm], bc)]
        st["n"] += m; st["chunks"] += 1; st["sec"] += time.time() - t1
        if st["chunks"] % 20 == 0 or st["n"] >= N:
            json.dump(st, open(fn + ".tmp", "w")); os.replace(fn + ".tmp", fn); print(f"t {t:g} n {st['n']:.3g} {st['sec']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
