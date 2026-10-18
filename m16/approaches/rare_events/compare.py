"""T340 POST HOC (DRI request, not pre-registered): equal-compute comparison per (model, event) with each run's executed C.
Sampling at C FPE (C inputs): relative RMS error sqrt((1 - p) / (p C)); when p C < 1 plain sampling usually returns 0, so
its typical relative error is taken as 1 (the zero answer), i.e. s = min(1, sqrt((1-p)/(pC))) if pC >= 1 else 1.
Ours: r = |est - p| / p. Win if r < s. -> results/compare.json, table"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "results", "score.json")))
out = []; print(f"{'event':10s} {'model':6s} {'var':3s} {'p':>9s} {'C':>8s} {'p*C':>8s} {'sampling rel err':>16s} {'ours':>7s} win")
for r in rows:
    if r["event"] == "E1(0.0)":
        continue
    p, C = r["truth"], r["C"]; pc = p * C
    s = np.sqrt((1 - p) / pc) if pc >= 1 else 1.0
    ours = abs(r["est"] - p) / p
    res = "tie" if abs(ours - s) < 0.05 else ("win" if ours < s else "loss")     # tie: both ~equally wrong (e.g. ~0 answers)
    out.append(dict(r, sampling_rel=s, ours_rel=ours, result=res))
    print(f"{r['event']:10s} {r['model']:6s} {r['variant']:3s} {p:9.2e} {C:8.0f} {pc:8.3g} {s:16.2f} {ours:7.2f} {res}")
for v in ("A", "B"):
    w = [o for o in out if o["variant"] == v]
    print(f"variant {v}: win {sum(o['result'] == 'win' for o in w)} / tie {sum(o['result'] == 'tie' for o in w)} / loss {sum(o['result'] == 'loss' for o in w)} of {len(w)}")
best = {}
for o in out:
    k = (o["event"], o["model"])
    if k not in best or o["ours_rel"] < best[k]["ours_rel"]:
        best[k] = o
print(f"better variant per (model, event) [itself post hoc]: win {sum(o['result'] == 'win' for o in best.values())} / tie {sum(o['result'] == 'tie' for o in best.values())} / loss {sum(o['result'] == 'loss' for o in best.values())} of {len(best)}")
json.dump(out, open(os.path.join(HERE, "results", "compare.json"), "w"), indent=1)
