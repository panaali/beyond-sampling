"""T192: estG parallel over the 2 T! true-boundary halves (independent sub-problems of estT.estT).

Workers run the unchanged estT_base geometry for one half and return its simplices; the parent integrates them with
estG.masses on `device` (cuda: GPU float64) and appends one JSON line per half to `log` (resume: halves already in the
log are skipped). Ops / STATS are additive over halves, so totals equal the sequential run.

usage: python estP.py H T seed mode [workers] [device] [eps]      (seed s or "s:noisy<i>" -> leaderboard._noisy)
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
    if "noisy" in name:
        from leaderboard import _noisy
        s, k = name.split(":noisy")
        return _noisy(zoo(H, T, int(s)), 4, 0.005, 7)[int(k)]
    return zoo(H, T, int(name))


def run(model, mode, workers=2, device="cpu", eps=0.1, log=None):
    estG.DEV[0] = device
    T = model[2].shape[0]
    if T not in B._HALVES:
        B._HALVES[T] = B.halves(T)
    nh = len(B._HALVES[T])
    done = {}
    if log and os.path.exists(log):
        for line in open(log):
            r = json.loads(line); done[r["half"]] = r
    todo = [i for i in range(nh) if i not in done]
    t0 = time.time(); tm = 0.0
    fo = open(log, "a") if log else None
    with mp.get_context("fork").Pool(workers, _init, (model, mode, eps)) as pool:
        for i, R, ops, st, sec in pool.imap_unordered(_half, todo):
            t1 = time.time(); err = 0.0; mops = 0; ns = 0
            for d, Rd in R.items():
                m, o = estG.masses(Rd, d); err += m; mops += o; ns += len(Rd)
            tm += time.time() - t1
            r = dict(half=i, err=err, ops=ops + mops, simplices=ns, sec_geometry=sec, **st)
            done[i] = r
            if fo:
                fo.write(json.dumps(r) + "\n"); fo.flush()
    if fo:
        fo.close()
    tot = dict(acc=1.0 - sum(done[i]["err"] for i in range(nh)), halves=nh)
    for k in ("ops", "simplices", "cells", "cut", "resolved", "sec_geometry"):
        tot[k] = sum(done[i][k] for i in range(nh))
    tot.update(sec_wall=time.time() - t0, sec_mass_parent=tm, workers=workers, device=device, resumed=nh - len(todo))
    return tot


if __name__ == "__main__":
    H, T, name, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    device = sys.argv[6] if len(sys.argv) > 6 else "cpu"
    eps = float(sys.argv[7]) if len(sys.argv) > 7 else 0.1
    out = os.path.join(HERE, "results", "runs"); os.makedirs(out, exist_ok=True)
    tag = f"H{H}T{T}_{name.replace(':', '_')}_{mode}"
    r = run(load_model(H, T, name), mode, workers, device, eps, os.path.join(out, tag + ".jsonl"))
    r.update(H=H, T=T, model=name, mode=mode, eps=eps)
    json.dump(r, open(os.path.join(out, tag + ".json"), "w"), indent=1)
    print(tag, json.dumps(r))
