"""T424 truth: own conditional-law MC for E1(t0) = {pick == position of rank 3, g23 > t0}, any T. SAMPLING (truth only).
Generalises rare_noise/truth.py (T371) from T = 10: condition on v3, the 3rd largest of T iid N(0,1):
  v3 ~ f(v) prop. to phi(v) Phi(v)^(T-3) (1 - Phi(v + t0))^2, the two top values iid N | > v3 + t0, the T-3 lows iid
  N | < v3, positions a uniform random permutation.  p = P_T(g23 > t0) x mean(pick == pos(v3)).
Stopping (pre-registered): chunks of CHUNK draws until >= 100 hits or N_MAX draws; a point is scored only with >= 50.
Diagnostics: hits with g12 < .1 (triple-tie share, T371's definition), hits with g23 > 6 (beyond the estimators' T_hi).
usage: python truth.py TAG T0 [N_MAX] -> results/truth/TAG@T0.json (checkpointed; resumes)"""
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.special import log_ndtr, ndtr, ndtri

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import models as M  # noqa: E402
import thresholds as TH  # noqa: E402

GRID = np.linspace(-9.0, 7.0, 3_200_001)
HITS_TARGET = 100
CHUNK = 200_000
BLK = 8192
ZERO_STOP = 1e9     # amendment 1 (README): stop a point with 0 hits after 1e9 draws (unscored either way)


def make_draw(T, t0):
    lf = (T - 3) * log_ndtr(GRID) - GRID ** 2 / 2 + 2 * log_ndtr(-(GRID + t0)); f = np.exp(lf - lf.max())
    c = np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) / 2 * np.diff(GRID))]); c /= c[-1]

    def draw(rng, n):
        v3 = np.interp(rng.random(n), c, GRID)
        top = -ndtri(rng.random((n, 2)) * ndtr(-(v3 + t0))[:, None])
        lows = ndtri(rng.random((n, T - 3)) * ndtr(v3)[:, None])
        vals = np.column_stack([top, v3, lows])
        pos = rng.permuted(np.tile(np.arange(T), (n, 1)), axis=1)
        x = np.empty_like(vals); x[np.arange(n)[:, None], pos] = vals
        return x, pos[:, 2], top.min(1) - v3, np.abs(top[:, 0] - top[:, 1])
    return draw


def pick(mod, xT):
    Wi, Wh, Wo = mod; h = torch.zeros(Wh.shape[0], xT.shape[1], dtype=torch.float64)
    for s in range(xT.shape[0]):
        h = torch.addmm(Wi * xT[s:s + 1], Wh, h).clamp_(min=0)
    return (Wo @ h).argmax(0).numpy()


def run(tag, t0, n_max=4e9):
    row = M.by_tag(tag); T = row["T"]; Wi, Wh, Wo = M.load(row)
    mod = tuple(torch.from_numpy(np.ascontiguousarray(w)) for w in (Wi[:, None], Wh, Wo))
    out_dir = os.environ.get("T424_TRUTH_DIR", os.path.join(HERE, "results", "truth"))   # working checkpoints
    os.makedirs(out_dir, exist_ok=True)
    fn = os.path.join(out_dir, f"{tag}@{t0:g}.json")
    st = json.load(open(fn)) if os.path.exists(fn) else dict(tag=tag, T=T, t0=t0, p_gap=TH.p_gap(T, t0), n=0, chunks=0, hits=0,
                                                             hits_g12_lt_p1=0, hits_g23_gt_6=0, sec=0.0,
                                                             seed=[424, row["H"], T, row["seed"], int(row["stream"] == "zoo"), int(round(100 * t0))])
    draw = make_draw(T, t0)
    while st["hits"] < HITS_TARGET and st["n"] < n_max and not (st["hits"] == 0 and st["n"] >= ZERO_STOP):
        t = time.time(); rng = np.random.default_rng(st["seed"] + [st["chunks"]])
        x, p3, g23, g12 = draw(rng, CHUNK); xT = torch.from_numpy(np.ascontiguousarray(x.T))
        hit = np.concatenate([pick(mod, xT[:, a:a + BLK]) for a in range(0, CHUNK, BLK)]) == p3
        st["hits"] += int(hit.sum()); st["hits_g12_lt_p1"] += int((hit & (g12 < 0.1)).sum()); st["hits_g23_gt_6"] += int((hit & (g23 > 6)).sum())
        st["n"] += CHUNK; st["chunks"] += 1; st["sec"] += time.time() - t
        if st["chunks"] % 25 == 0 or st["hits"] >= HITS_TARGET:
            st["p"] = st["p_gap"] * st["hits"] / st["n"]; json.dump(st, open(fn + ".tmp", "w"), indent=1); os.replace(fn + ".tmp", fn)
    st["p"] = st["p_gap"] * st["hits"] / st["n"]; st["rel_se"] = 1 / np.sqrt(max(st["hits"], 1))
    json.dump(st, open(fn + ".tmp", "w"), indent=1); os.replace(fn + ".tmp", fn)
    return st


if __name__ == "__main__":
    torch.set_num_threads(1)
    run(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 4e9)
