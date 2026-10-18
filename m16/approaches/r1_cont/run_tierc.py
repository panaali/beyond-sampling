"""T330 tier C on all 80 DEV2: C1/C2 x lines {non-blind, all 3}; declared error d = est - T317e, cost, error vs truth,
and the DRI's firing fraction on the non-blind lines (skipped later cells / oracle-skippable later cells).
-> results_tierc.json"""
import json
import os
import sys

import numpy as np

import cont_e as E
import floors as FL
import screen_c as S
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r0_integral"))
import r0  # noqa: E402

CFG = {f"{mo}_{ln}": dict(mode=mo, lines=L) for mo in ("C1", "C2")
       for ln, L in (("nb", ("nb-23", "nb-12")), ("all", ("nb-23", "nb-12", "b-23")))}
truth = r0.truths()["truths"]; rows = {}
for nm, m in r0.dev2():
    e0, f0, info0, _ = S.est(m, parts=True, lines=())
    model = tuple(np.asarray(w, np.float64) for w in m); k = E.TRI[:, 1]; ideal = {}
    for tb in E.TABLES:
        for kind in ("23", "12"):
            if f"line{kind}" in tb:
                x, dx = tb[f"line{kind}"]; _, _, cells = FL.walk_rec(model, x, dx, k, E.TMAX); lw = {}
                for r_, c_, w, _ in cells:
                    if w:
                        lw[r_] = max(lw.get(r_, -1), c_)
                ideal[f"{tb['name']}-{kind}"] = sum(max(v, 0) for v in lw.values())
    rows[nm] = dict(est0=e0, C0=f0 / 5948, truth=truth[nm]["acc"], base_restarts={l: v["restarts"] for l, v in info0.items()},
                    ideal=ideal)
    for c, kw in CFG.items():
        e, f, info, _ = S.est(m, parts=True, **kw)
        rows[nm][c] = dict(est=e, d_pp=100 * (e - e0), C=f / 5948, restarts={l: v["restarts"] for l, v in info.items()})
    print(nm, {c: (round(rows[nm][c]["C"]), round(rows[nm][c]["d_pp"], 4)) for c in CFG}, flush=True)
R = list(rows.values()); p = np.array([r["truth"] for r in R]); v = float(np.mean(p * (1 - p)))
def stats(key):
    e = np.array([(r[key]["est"] if key else r["est0"]) - r["truth"] for r in R]); Cm = float(np.mean([r[key]["C"] if key else r["C0"] for r in R]))
    rms = float(np.sqrt((e ** 2).mean()))
    return dict(C=Cm, rms_pp=100 * rms, bias_pp=100 * float(e.mean()), F_len=rms ** 2 * Cm / v, F_strict=rms ** 2 * max(Cm, 1000) / .074)
s = {"T317e": stats(None)}
for c in CFG:
    d = np.array([r[c]["d_pp"] for r in R])
    fire = []
    for l in ("nb-23", "nb-12"):
        b = sum(r["base_restarts"][l] for r in R); sc = sum(r[c]["restarts"][l] for r in R); idl = sum(r["ideal"][l] for r in R)
        fire.append((b - sc) / max(b - idl, 1))
    s[c] = dict(stats(c), declared_rms_pp=float(np.sqrt((d ** 2).mean())), declared_mean_pp=float(d.mean()),
                declared_max_pp=float(np.abs(d).max()), fire_frac_nb23=fire[0], fire_frac_nb12=fire[1])
    s[c]["precheck_go"] = bool(s[c]["declared_rms_pp"] <= .10 and min(fire) >= .70)
    s[c]["precheck_stop"] = bool(s[c]["declared_rms_pp"] >= .25)
    s[c]["tierC_go"] = bool(s[c]["declared_rms_pp"] <= .10 and s[c]["C"] <= 2500)
print(json.dumps(s, indent=1))
json.dump(dict(summary=s, rows=rows), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_tierc.json"), "w"), indent=1)
