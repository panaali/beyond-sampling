"""T370 stage 2 sealed truths (SAMPLING, truth only): conditional-law MC for the 8 selected pairs (results/t370s2_selection.json).
E1(t0): x | g23 > t0 (truthc.sampler), hit = pick == p3; p = P(g23 > t0) x hit rate.
E2(t):  x | g12 >= t, g23 >= t (pre-registered analogue: v2 ~ 90 phi(v2)(1 - Phi(v2 + t)) Phi(v2 - t)^8, v1 > v2 + t,
        v3 < v2 - t, lows | < v3), hit = pick != p2; p = P(both gaps >= t) x hit rate.
1e9 conditional samples per pair; under the DRI amendment a pair with < 50 hits is extended by 10x steps up to 1e11.
Full numbers go only to results/t370s2_truth_sealed.json; stdout shows only "hits >= 50: yes / no" per pair.
usage: python stage2_truth.py [N] (default 1e9; run again with a larger N to extend, the pass continues from saved n)"""
import json
import os
import sys
import time

import numpy as np
from scipy.special import ndtr, ndtri

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from truthc import GRID, sampler, p_gap  # noqa: E402

OUT = os.path.join(HERE, "results", "t370s2_truth_sealed.json")
N = int(float(sys.argv[1])) if len(sys.argv) > 1 else 10 ** 9
CHUNK = 500_000; BLOCK = 50 * CHUNK


def e2_sampler(t):
    f = 90 * np.exp(-GRID ** 2 / 2) / np.sqrt(2 * np.pi) * ndtr(-(GRID + t)) * ndtr(GRID - t) ** 8
    c = np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) / 2 * np.diff(GRID))]); P = float(c[-1]); c /= P

    def draw(rng, n):
        v2 = np.interp(rng.random(n), c, GRID)
        v1 = -ndtri(rng.random(n) * ndtr(-(v2 + t)))
        v3 = ndtri(ndtr(v2 - t) * rng.random(n) ** (1 / 8))
        lows = ndtri(ndtr(v3)[:, None] * rng.random((n, 7)))
        vals = np.column_stack([v1, v2, v3, lows]); perm = np.argsort(rng.random((n, 10)), 1)
        x = np.empty_like(vals); x[np.arange(n)[:, None], perm] = vals
        return x, perm[:, 1]
    return draw, P


def picks(m, x):
    Wi, Wh, Wo = m; h = np.zeros((len(x), 16)); WhT = Wh.T.copy()
    for s in range(10):
        h = h @ WhT; h += x[:, s:s + 1] * Wi; np.maximum(h, 0, out=h)
    return (h @ Wo.T).argmax(1)


z = np.load(os.path.join(HERE, "ext", "dev3", "weights.npz")); names = list(map(str, z["names"]))
sel = json.load(open(os.path.join(HERE, "results", "t370s2_selection.json")))["selected"]
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
for c in sel:
    key = f"{c['model']}:{c['event']}:{c['t']}"; r = out.get(key, dict(n=0, hits=0, sec=0.0))
    if r["n"] >= N:
        print(key, "hits >= 50:", "yes" if r["hits"] >= 50 else "no", flush=True); continue
    i = names.index(c["model"]); m = tuple(z[k][i].astype(np.float64) for k in ("Wi", "Wh", "Wo"))
    if c["event"] == "E1":
        draw, P = sampler(c["t"]), p_gap(c["t"])
    else:
        draw, P = e2_sampler(c["t"])
    # block b (2.5e7 samples) uses its own fixed stream [370200, model, event, 100 t, b]: resumes and extensions are exact
    # and never reuse samples
    assert N % BLOCK == 0 and r["n"] % BLOCK == 0
    for b in range(r["n"] // BLOCK, N // BLOCK):
        t0 = time.time(); rng = np.random.default_rng([370_200, int(c["model"][6:]), int(c["event"][1]), int(round(100 * c["t"])), b])
        for _ in range(BLOCK // CHUNK):
            x, pos = draw(rng, CHUNK); y = picks(m, x)
            r["hits"] += int(((y == pos) if c["event"] == "E1" else (y != pos)).sum())
        r["n"] += BLOCK; r["sec"] += time.time() - t0; q = r["hits"] / r["n"]
        r.update(P_cond=P, rate=q, p=P * q, se=P * float(np.sqrt(max(q * (1 - q), 1e-300) / r["n"])),
                 seed_note="block b: default_rng([370200, model, event, 100 t, b]), block 2.5e7")
        out[key] = r; json.dump(out, open(OUT, "w"), indent=1)
    print(key, "hits >= 50:", "yes" if r["hits"] >= 50 else "no", flush=True)
