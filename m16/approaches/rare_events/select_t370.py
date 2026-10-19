"""T370 stage 2 truth-blind selection (rule pre-registered 2026-10-18 23:40 + DRI amendment 2026-10-19 13:52).
Inputs: results/t370s2/{k3,q16e1,q16e2}_<model>.json (all 16 models; q16 files from either session, copied verbatim).
d = |log(Q16 / K3)| per candidate; top 4 E1 and top 4 E2 by d, at most 2 pairs per model (over E1 + E2 together),
ties broken by model name. Implemented as one pass over all 64 in decreasing d, so the per-model cap does not favour
either event type (fixed here, before any Q16 estimate is read). The full ranking is kept, for replacements under the >= 50 hits rule.
-> results/t370s2_selection.json"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results", "t370s2")
MODELS = ["dev3_s%d" % s for s in range(548, 564)]
EV = {"E1": ("q16e1", (2.35, 2.75)), "E2": ("q16e2", (1.25, 1.5))}
cand = []
for m in MODELS:
    k3 = json.load(open(os.path.join(R, f"k3_{m}.json")))
    for ev, (mode, ts) in EV.items():
        q = json.load(open(os.path.join(R, f"{mode}_{m}.json")))
        for t in ts:
            a, b = q[ev][str(t)], k3[ev][str(t)]
            d = abs(np.log(a / b)) if a > 0 and b > 0 else float("inf")
            cand.append(dict(model=m, event=ev, t=t, q16=a, k3=b, d=float(d)))
assert len(cand) == 64
rank = sorted(cand, key=lambda c: (-c["d"], c["model"], c["event"], c["t"]))
per = {}; n = {"E1": 0, "E2": 0}; sel = []
for c in rank:                                       # one pass in decreasing d: quota 4 per event, cap 2 per model
    if n[c["event"]] == 4 or per.get(c["model"], 0) == 2:
        continue
    sel.append(c); per[c["model"]] = per.get(c["model"], 0) + 1; n[c["event"]] += 1
json.dump(dict(rule="T370 stage 2, README 2026-10-18 23:40 + 2026-10-19 13:52 amendment", selected=sel, ranking=rank),
          open(os.path.join(HERE, "results", "t370s2_selection.json"), "w"), indent=1)
for c in sel:
    print(c)
