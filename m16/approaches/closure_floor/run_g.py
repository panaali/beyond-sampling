"""T339 run: T317f (s=0, checked bitwise against the frozen copy) and the spread patterns s = +1, -1 on all 80 DEV2.
Pre-registered selection: min DEV2 MSE x C, one freeze, only if below T317f. -> results/t339.json"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "cheap_frontier"))
import cf  # noqa: E402
import cont_f_t317f as F  # noqa: E402
import cont_g as G  # noqa: E402

ms = cf.dev2_models(); tr = cf.dev2_truths()
TAB = {0.0: G.TABLES, 1.0: G._tables(1.0), -1.0: G._tables(-1.0)}
rows = {}; bitwise = True; nodes = {}
for name, m in ms.items():
    ef, ff = F.est(m); rows[name] = {}
    for s, tab in TAB.items():
        st = {}; e, f = G.est(m, tables=tab, stats=st)
        if s == 0.0:
            bitwise &= (e == ef) and (f == ff)
        rows[name][str(s)] = dict(est=e, C=f / cf.FPE); nodes.setdefault(str(s), set()).update(st.values())
pq = np.mean([tr[n]["acc"] * (1 - tr[n]["acc"]) for n in ms])
summ = {}
for s in ("0.0", "1.0", "-1.0"):
    e = np.array([rows[n][s]["est"] - tr[n]["acc"] for n in ms]); C = np.mean([rows[n][s]["C"] for n in ms])
    mse = float((e ** 2).mean() - np.mean([tr[n]["se"] ** 2 for n in ms]))
    summ[s] = dict(rms_pp=100 * np.sqrt(mse), bias_pp=100 * e.mean(), C=C, F=mse * max(C, 1000) / pq,
                   mse_x_C=1e4 * mse * C, nodes_per_line=sorted(nodes[s]),
                   C_range=[float(min(rows[n][s]["C"] for n in ms)), float(max(rows[n][s]["C"] for n in ms))])
best = min(("1.0", "-1.0"), key=lambda s: summ[s]["mse_x_C"])
sel = best if summ[best]["mse_x_C"] < summ["0.0"]["mse_x_C"] else None
out = dict(bitwise_s0_equals_T317f=bool(bitwise), summary=summ, selected=sel, rows=rows)
json.dump(out, open(os.path.join(HERE, "results", "t339.json"), "w"), indent=1)
print("s=0 bitwise = T317f on all 80:", bitwise)
for s, v in summ.items():
    print(f"s={s:5s} RMS {v['rms_pp']:.3f} bias {v['bias_pp']:+.3f} C {v['C']:7.0f} [{v['C_range'][0]:.0f},{v['C_range'][1]:.0f}]"
          f" F {v['F']:.2f} MSExC {v['mse_x_C']:.0f} tree nodes/line {v['nodes_per_line']}")
print("selected:", sel)
