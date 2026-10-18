"""T371 truth: own conditional-law MC for E1(t0) = {pick == position of rank 3, g23 > t0}. SAMPLING (truth only).
Construction (differs from rare_events/truthc.py, which draws v2 first): condition on v3, the 3rd largest of 10 iid N(0,1).
  v3 ~ f(v) prop. to 360 phi(v) Phi(v)^7 (1 - Phi(v + t0))^2   (= density of rank 3 times P(both top values > v + t0 | > v)),
  the two top values iid N | > v3 + t0, the 7 lows iid N | < v3, positions a uniform random permutation.
P(g23 > t0) = int f (quadrature on the same grid). p = P(g23 > t0) x mean(pick == pos(v3)), exact up to the grid inverse CDF.
The same draws are pushed through every model of the panel (common random numbers; each model's estimate is unbiased).
Diagnostics per model: hits with g23 > 3.5 (mass beyond the estimator's T_hi) and hits with g12 < .1 (triple-tie share).
usage: python truth.py T0 N [CHUNK] -> results/truth_T0.json (checkpointed; resumes)"""
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

GRID = np.linspace(-9.0, 7.0, 3_200_001)


def logf(t0):
    return np.log(360.0) - GRID ** 2 / 2 - 0.5 * np.log(2 * np.pi) + 7 * log_ndtr(GRID) + 2 * log_ndtr(-(GRID + t0))


def p_gap(t0):
    f = np.exp(logf(t0)); return float(((f[1:] + f[:-1]) / 2 * np.diff(GRID)).sum())


def make_draw(t0):
    f = np.exp(logf(t0)); c = np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) / 2 * np.diff(GRID))]); c /= c[-1]
    base = np.tile(np.arange(10), (1, 1))

    def draw(rng, n):
        v3 = np.interp(rng.random(n), c, GRID)
        top = -ndtri(rng.random((n, 2)) * ndtr(-(v3 + t0))[:, None])           # iid N | > v3 + t0
        lows = ndtri(rng.random((n, 7)) * ndtr(v3)[:, None])                   # iid N | < v3
        vals = np.column_stack([top, v3, lows])                                # column 2 = v3
        pos = rng.permuted(np.repeat(base, n, 0), axis=1)                      # pos[:, c] = position of column c
        x = np.empty_like(vals); x[np.arange(n)[:, None], pos] = vals
        g23 = top.min(1) - v3; g12 = np.abs(top[:, 0] - top[:, 1])
        return x, pos[:, 2], g23, g12
    return draw


BLK = 8192


def pick(mod, xT):
    """float64 forward in (H, n) layout (cache-friendly); returns argmax of logits per column."""
    Wi, Wh, Wo = mod; h = torch.zeros(Wh.shape[0], xT.shape[1], dtype=torch.float64)
    for s in range(10):
        h = torch.addmm(Wi * xT[s:s + 1], Wh, h).clamp_(min=0)
    return (Wo @ h).argmax(0).numpy()


def main():
    t0 = float(sys.argv[1]); N = int(float(sys.argv[2])); chunk = int(float(sys.argv[3])) if len(sys.argv) > 3 else 250_000
    P = MP.panel(); names = sorted(P)
    torch.set_num_threads(1)
    Ms = [tuple(torch.from_numpy(np.ascontiguousarray(w)) for w in (P[m][0][:, None], P[m][1], P[m][2])) for m in names]
    fn = os.path.join(HERE, "results", f"truth_{t0:g}.json")
    st = json.load(open(fn)) if os.path.exists(fn) else dict(t0=t0, p_gap=p_gap(t0), n=0, chunks=0, seed=371_000 + int(round(100 * t0)),
                                                             hits={m: 0 for m in names}, hits_g23_gt_3p5={m: 0 for m in names},
                                                             hits_g12_lt_p1={m: 0 for m in names}, sec=0.0)
    draw = make_draw(t0)
    while st["n"] < N:
        t = time.time(); m = min(chunk, N - st["n"])
        rng = np.random.default_rng([st["seed"], st["chunks"]])                 # one stream per chunk: resumable
        x, p3, g23, g12 = draw(rng, m); far = g23 > 3.5; tie = g12 < 0.1; xT = torch.from_numpy(np.ascontiguousarray(x.T))
        for nm, mod in zip(names, Ms):
            hit = np.concatenate([pick(mod, xT[:, a:a + BLK]) for a in range(0, m, BLK)]) == p3
            st["hits"][nm] += int(hit.sum()); st["hits_g23_gt_3p5"][nm] += int((hit & far).sum()); st["hits_g12_lt_p1"][nm] += int((hit & tie).sum())
        st["n"] += m; st["chunks"] += 1; st["sec"] += time.time() - t
        if st["chunks"] % 20 == 0 or st["n"] >= N:
            json.dump(st, open(fn + ".tmp", "w"), indent=1); os.replace(fn + ".tmp", fn)
            print(f"t0 {t0:g} n {st['n']:.3g} {st['sec']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
