"""Score the frozen estimators at L = 32 against long-MC truths (pre-registered in README, 'L = 32').
usage: python score32.py MODEL -> appends data/t446/score32.jsonl"""
import glob
import json
import os
import sys
import time
import zlib

import numpy as np

from est import METHODS
from s3lib import DATA, Dyck2, load
from taus import GRID

name = sys.argv[1]
fs = sorted(glob.glob(f"{DATA}/{name}_L32_mc_*.npz"))
mx = np.concatenate([np.load(f)["maxes"] for f in fs]); N = sum(int(np.load(f)["N"]) for f in fs)
cnt = np.array([(mx > t).sum() for t in GRID])
B = cnt / N
ok = (cnt >= 100) & (B <= 1e-5)
if not ok.any():
    print(name, "no tau with >= 100 MC events and B <= 1e-5"); sys.exit()
i = int(np.argmin(np.where(ok, np.abs(np.log10(np.maximum(B, 1e-300)) - np.log10(5e-6)), np.inf)))
tau, truth, n_ev = float(GRID[i]), float(B[i]), int(cnt[i])
plan = [(1e5, "ce", 5), (1e5, "top", 5), (1e5, "amsce", 3), (1e6, "ce", 3), (1e6, "top", 3)]
done = set()
if os.path.exists(f"{DATA}/score32.jsonl"):
    for l in open(f"{DATA}/score32.jsonl"):
        r = json.loads(l); done.add((r["model"], r["budget"], r["method"], r["rep"]))
model, _ = load(name)
lang = Dyck2(L=32)
out = open(f"{DATA}/score32.jsonl", "a")
for budget, meth, reps in plan:
    for r in range(reps):
        if (name, budget, meth, r) in done:
            continue
        seed = zlib.crc32(f"{name}|L32|{budget:.0e}|{meth}|{r}".encode())
        t0 = time.time()
        e, c = METHODS[meth](model, lang, 32, tau, budget, np.random.default_rng(seed))
        rec = dict(model=name, tau=tau, truth=truth, truth_events=n_ev, truth_N=N, budget=budget, method=meth, rep=r,
                   est=e, cost=c, sec=time.time() - t0)
        out.write(json.dumps(rec) + "\n"); out.flush()
        print(name, f"tau {tau:.4f} B {truth:.2e} ({n_ev} ev) {meth} {budget:.0e} r{r}: ratio {e / truth:.3f} cost {c:.0f} {time.time() - t0:.0f}s", flush=True)
