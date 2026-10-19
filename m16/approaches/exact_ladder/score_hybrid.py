"""T46 arm B scoring (pre-registered, README 'T46 resumed'): hybrid at tau = 1e-2 vs (i) sampling at the executed
cost and (ii) the frozen walk alone at the hybrid's executed cost, 17 fixed rule seeds, B = 16k, 64k.
usage: python score_hybrid.py T [tau]   -> appends raw rows to score_T{T}.jsonl (resumable)"""
import json
import os
import sys
import time

import numpy as np

sys.path.append("../..")
import leaderboard as L  # noqa: E402
import hybrid as Hy  # noqa: E402

FE = Hy.FE
T = int(sys.argv[1]); tau = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-2
RULES = [FE.RULE_SEED] + [FE.RULE_SEED + 1000 * i for i in range(1, 17)]
BUDGETS = (16_000, 64_000)
out = f"score_T{T}.jsonl"
done = set()
if os.path.exists(out):
    for l in open(out):
        d = json.loads(l); done.add((d["name"], d["B"], d["rule"]))
ms = L.models(16, T)
for name, m in ms.items():
    if all((name, B, r) in done for B in BUDGETS for r in RULES):
        continue
    t0 = time.process_time()
    ex = Hy.exact_part(m, tau)
    tex = time.process_time() - t0
    for B in BUDGETS:
        for r in RULES:
            if (name, B, r) in done:
                continue
            t1 = time.process_time()
            rest, used = Hy.walk_rest(m, B, ex["keys"], r)
            th = time.process_time() - t1
            cost = ex["fpe"] + used
            t1 = time.process_time()
            aw, cw = FE.est(m, cost, None, rule_seed=r)
            tw = time.process_time() - t1
            row = dict(name=name, T=T, tau=tau, B=B, rule=r, hybrid=1 - ex["wrong"] - rest, hybrid_cost=cost,
                       exact_fpe=ex["fpe"], exact_wrong=ex["wrong"], exact_err=ex["err"], cells=len(ex["keys"]),
                       massE=ex["mass"], walk_rest=rest, walk_rest_fpe=used, walk=aw, walk_cost=cw,
                       cpu_exact=tex, cpu_hybrid_walk=th, cpu_walk=tw)
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
    print(name, len(ex["keys"]), f"{ex['fpe']:.3g}", f"{tex:.0f}s", flush=True)
