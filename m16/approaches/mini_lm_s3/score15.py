"""Score the frozen estimators at L = 15 against exact truths. Appends to data/t446/score15.jsonl.
usage: python score15.py MODEL [budgets] [methods]"""
import json
import os
import sys
import time
import zlib

import numpy as np

from est import METHODS
from s3lib import DATA, Dyck2, load
from taus import points

name = sys.argv[1]
plan = [(1e5, "ce", 5), (1e5, "top", 5), (1e5, "ams", 3), (1e5, "amsl", 3), (1e5, "amsce", 3),
        (1e4, "ce", 3), (1e4, "top", 3), (1e6, "ce", 3), (1e6, "top", 3)]
if os.environ.get("S3_PLAN") == "sealed":  # pre-registered sealed MLP1 test: 1e5 only
    plan = [(1e5, "ce", 3), (1e5, "top", 3)]
if len(sys.argv) > 2:
    plan = [p for p in plan if p[1] in sys.argv[2].split(",")]
model, _ = load(name)
lang = Dyck2(L=15)
done = set()
if os.path.exists(f"{DATA}/score15.jsonl"):
    for l in open(f"{DATA}/score15.jsonl"):
        r = json.loads(l); done.add((r["model"], round(r["tau"], 6), r["budget"], r["method"], r["rep"]))
out = open(f"{DATA}/score15.jsonl", "a")
for tau, truth, target in points(name):
    for budget, meth, reps in plan:
        for r in range(reps):
            if (name, round(tau, 6), budget, meth, r) in done:
                continue
            seed = zlib.crc32(f"{name}|{tau:.6f}|{budget:.0e}|{meth}|{r}".encode())
            t0 = time.time()
            e, c = METHODS[meth](model, lang, 15, tau, budget, np.random.default_rng(seed))
            rec = dict(model=name, tau=tau, truth=truth, target=target, budget=budget, method=meth, rep=r,
                       est=e, cost=c, sec=time.time() - t0)
            out.write(json.dumps(rec) + "\n"); out.flush()
            print(name, f"tau {tau:.4f} B {truth:.2e} {meth} {budget:.0e} r{r}: ratio {e / truth:.3f} cost {c:.0f} {time.time() - t0:.0f}s", flush=True)
