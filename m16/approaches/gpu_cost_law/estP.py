"""T192: estG parallel over the 2 T! true-boundary halves (independent sub-problems of estT.estT).

Workers run the unchanged estT_base geometry for one half and return its simplices; the parent integrates them with
estG.masses on `device` (cuda: GPU float64) and appends one JSON line per half to `log` (resume: halves already in the
log are skipped). Ops / STATS are additive over halves, so totals equal the sequential run.

usage: python estP.py H T seed mode [workers] [device] [eps]      (seed s, "s:noisy<i>" -> _noisy(zoo s, 4, .005, 7)[i], "s:fresh<k>" -> _noisy(zoo s, 1, .005, k)[0])
       -> results/runs/H{H}T{T}_{name}_{mode}.jsonl (per half) and .json (summary)
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
U = os.path.join(HERE, "..", "understand_m43")
sys.path[:0] = [HERE, U, os.path.join(HERE, "..", "arc_m43_repro"), os.path.join(HERE, "..", "..")]
import estG  # noqa: E402
import estT_base as B  # noqa: E402
from m43 import Ops  # noqa: E402
if os.environ.get("T192_FAST", "1") == "1":      # numba geometry hot spots (geomfast); 0 = pure estT_base
    import geomfast  # noqa: E402,F401

_W = {}


def _collect():
    """Worker-side replacement of the buffer: keep simplices per dimension."""
    class Col:
        def __init__(self):
            self.L = {}

        def add(self, R):
            self.L.setdefault(R.shape[0], []).append(R.copy())
    return Col()


def _init(model, mode, eps):
    _W.update(model=model, mode=mode, eps=eps)
    T = model[2].shape[0]
    if T not in B._HALVES:
        B._HALVES[T] = B.halves(T)


def _half(i):
    model, mode, eps = _W["model"], _W["mode"], _W["eps"]
    Wi, Wh, Wo = model; T = Wo.shape[0]
    for k in B.STATS:
        B.STATS[k] = 0
    exact = mode == "exact"; B.MODE[0] = mode
    estG.BUF[0] = _collect(); o = Ops(); t0 = time.time()
    h = B._HALVES[T][i]
    g1 = (h["sigma"] * Wi > 0).astype(float); o.add(len(Wi))
    A1 = np.outer(Wi, np.eye(T)[0])
    e0 = B._finish(h["cone"], g1, A1, 1, model, h["truth"], h["n_true"], eps, o, exact)
    assert e0 == 0.0
    R = {d: np.array(v) for d, v in estG.BUF[0].L.items()}
    st = {k: v for k, v in B.STATS.items() if k != "simplices"}
    return i, R, o.n, st, time.time() - t0


def load_model(H, T, name):
    from zoo44 import zoo
    if "fresh" in name:                       # T193: fresh noise seed, one copy, sigma as estT_fix
        from leaderboard import _noisy
        s, k = name.split(":fresh")
        return _noisy(zoo(H, T, int(s)), 1, 0.005, int(k))[0]
    if "noisy" in name:
        from leaderboard import _noisy
        s, k = name.split(":noisy")
        return _noisy(zoo(H, T, int(s)), 4, 0.005, 7)[int(k)]
    return zoo(H, T, int(name))


def run(model, mode, workers=2, device="cpu", eps=0.1, log=None, sample=None, nomass=False):
    estG.DEV[0] = device
    T = model[2].shape[0]
    if T not in B._HALVES:
        B._HALVES[T] = B.halves(T)
    nh = len(B._HALVES[T])
    done = {}
    if log and os.path.exists(log):
        for line in open(log):
            r = json.loads(line); done[r["half"]] = r
    idx = list(range(nh))
    if sample:                                  # (frac, seed): simple random sample of halves, for cost estimates
        idx = sorted(np.random.default_rng(sample[1]).choice(nh, max(1, round(sample[0] * nh)), replace=False).tolist())
    todo = [i for i in idx if i not in done]
    t0 = time.time(); tm = 0.0
    fo = open(log, "a") if log else None
    with mp.get_context("fork").Pool(workers, _init, (model, mode, eps)) as pool:
        for i, R, ops, st, sec in pool.imap_unordered(_half, todo):
            t1 = time.time(); err = 0.0; mops = 0; ns = 0
            nthin = 0
            for d, Rd in R.items():
                ns += len(Rd); nthin += int((np.abs(np.linalg.det(Rd)) < 1e-6).sum())
                if not nomass:
                    m, o = estG.masses(Rd, d); err += m; mops += o
            tm += time.time() - t1
            r = dict(half=i, err=None if nomass else err, ops=ops + mops, ops_geometry=ops, simplices=ns, thin=nthin,
                     sec_geometry=sec, **st)
            done[i] = r
            if fo:
                fo.write(json.dumps(r) + "\n"); fo.flush()
    if fo:
        fo.close()
    full = not sample and not nomass
    tot = dict(acc=1.0 - sum(done[i]["err"] for i in range(nh)) if full else None, halves=nh, halves_run=len(idx))
    for k in ("ops", "ops_geometry", "simplices", "thin", "cells", "cut", "resolved", "sec_geometry"):
        v = np.array([done[i].get(k, np.nan) for i in idx], float)
        tot[k] = float(v.sum()) if not sample else float(v.mean() * nh)       # sample: estimated total
        if sample and len(v) > 1:
            tot[k + "_se"] = float(v.std(ddof=1) * nh / np.sqrt(len(v)) * np.sqrt(1 - len(v) / nh))
    tot.update(sec_wall=time.time() - t0, sec_mass_parent=tm, workers=workers, device=device, resumed=nh - len(todo))
    return tot


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("H", type=int); ap.add_argument("T", type=int); ap.add_argument("name"); ap.add_argument("mode")
    ap.add_argument("workers", type=int, nargs="?", default=2); ap.add_argument("device", nargs="?", default="cpu")
    ap.add_argument("eps", type=float, nargs="?", default=0.1)
    ap.add_argument("--sample", type=float, default=None, help="fraction of halves (seeded random sample)")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--nomass", action="store_true", help="geometry only: count simplices, no mass values")
    a = ap.parse_args()
    out = os.path.join(HERE, "results", "runs"); os.makedirs(out, exist_ok=True)
    tag = f"H{a.H}T{a.T}_{a.name.replace(':', '_')}_{a.mode}" + ("_nomass" if a.nomass else "") + \
        (f"_s{a.sample:g}_{a.sample_seed}" if a.sample else "")
    r = run(load_model(a.H, a.T, a.name), a.mode, a.workers, a.device, a.eps, os.path.join(out, tag + ".jsonl"),
            (a.sample, a.sample_seed) if a.sample else None, a.nomass)
    r.update(H=a.H, T=a.T, model=a.name, mode=a.mode, eps=a.eps, fast=os.environ.get("T192_FAST", "1"))
    json.dump(r, open(os.path.join(out, tag + ".json"), "w"), indent=1)
    print(tag, json.dumps(r))
