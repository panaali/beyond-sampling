"""T330b gate on all 80 DEV2: the tier-B v3 screen (screen_d, norm track) at its default (try at every flip) and at the cheapest
schedule found on 4 models (dmax .05, one try per row). Soundness audit: per-row line profiles equal to T317e's
(<= 1e-12), plus max |est - T317e|. -> results_t330b.json"""
import json
import os
import sys

import numpy as np

import cont_e as E
import screen_c as S0
import screen_d as S
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r0_integral"))
import r0  # noqa: E402

CFG = {"every_flip": dict(), "dmax.05_once": dict(dmax=0.05, maxtry=1)}
rows = {}
for nm, m in r0.dev2():
    e0, f0 = E.est(m); _, _, _, p0 = S0.est(m, parts=True, lines=())
    rows[nm] = dict(est0=e0, C0=f0 / 5948)
    for c, kw in CFG.items():
        e, f, info, p = S.est(m, parts=True, **kw)
        rows[nm][c] = dict(d_pp=100 * (e - e0), C=f / 5948, prof_maxdiff=max(float(np.abs(p[l] - p0[l]).max()) for l in p),
                           stopped={l: v["stopped"] for l, v in info.items()}, tried={l: v["tried"] for l, v in info.items()})
    print(nm, {c: (round(rows[nm][c]["C"]), rows[nm][c]["d_pp"]) for c in CFG}, round(f0 / 5948), flush=True)
s = {"C_T317e": float(np.mean([r["C0"] for r in rows.values()]))}
for c in CFG:
    s[c] = dict(C=float(np.mean([r[c]["C"] for r in rows.values()])), max_abs_d_pp=max(abs(r[c]["d_pp"]) for r in rows.values()),
                max_prof_diff=max(r[c]["prof_maxdiff"] for r in rows.values()))
    s[c]["go"] = bool(s[c]["max_abs_d_pp"] <= 1e-4 and s[c]["C"] <= 3000)
print(json.dumps(s, indent=1))
json.dump(dict(summary=s, rows=rows), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_t330b.json"), "w"), indent=1)
