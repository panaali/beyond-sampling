"""Blind estimates (no truths read): frozen est at 4k/16k/64k FPE for the frozen rule seed and 8 alternative rule
seeds, plus the 1M-FPE validation run (frozen rule seed). -> est.jsonl
usage: python run.py [worker_index n_workers]"""
import sys
import time

import common as Cm

E = Cm.frozen(compiled=True)
RULES = [31415] + [31415 + 101 * 1000 * k for k in range(1, 9)]
JOBS = [(31415, b) for b in (4000, 16000, 64000, 1_000_000)] + [(s, b) for s in RULES[1:] for b in (4000, 16000, 64000)]
wi, nw = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
done = {(r["tag"], r["rule"], r["budget"]) for r in Cm.jl_read("est.jsonl")}
models = sorted(Cm.registry(), key=lambda r: (r["H"] ** 2 * r["T"], r["tag"]))[wi::nw]
for r in models:
    m = None
    for s, b in JOBS:
        if (r["tag"], s, b) in done:
            continue
        m = m or Cm.load(r["tag"])
        for k in Cm.STATS:
            Cm.STATS[k] = 0
        t0 = time.time(); e, c = E.est(m, b, rule_seed=s)
        row = {"tag": r["tag"], "H": r["H"], "T": r["T"], "rule": s, "budget": b, "est": e, "cost": c,
               **Cm.STATS, "secs": round(time.time() - t0, 2)}
        Cm.jl_add("est.jsonl", row); print(row, flush=True)
