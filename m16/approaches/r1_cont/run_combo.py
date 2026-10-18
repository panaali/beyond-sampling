"""T331+C1 combination on all 80 DEV2 (README pre-registration): tmax 1 + zero-cost extension + tier C1 on all lines.
Reports bias, RMS, C, C1's declared error (vs the same variant without C1 = T331 (iv)); go: MSE x C <= .70 x T317e's.
-> results_combo.json"""
import json
import os
import sys

import numpy as np

import cont_e as E
import screen_c as S
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r0_integral"))
import r0  # noqa: E402

ALL = ("nb-23", "nb-12", "b-23")
truth = r0.truths()["truths"]; rows = {}
for nm, m in r0.dev2():
    e0, f0 = E.est(m)
    e4, f4 = S.est(m, lines=(), tmax=1.0, ext=True)
    ec, fc, info, _ = S.est(m, parts=True, lines=ALL, mode="C1", tmax=1.0, ext=True)
    rows[nm] = dict(truth=truth[nm]["acc"], T317e=(e0, f0 / 5948), iv=(e4, f4 / 5948), combo=(ec, fc / 5948),
                    declared_pp=100 * (ec - e4), stopped={l: v["stopped"] for l, v in info.items()})
    print(nm, {k: (round(100 * (rows[nm][k][0] - rows[nm]["truth"]), 3), round(rows[nm][k][1])) for k in ("T317e", "iv", "combo")},
          round(rows[nm]["declared_pp"], 4), flush=True)
R = list(rows.values()); p = np.array([r["truth"] for r in R]); var = float(np.mean(p * (1 - p))); s = {}
for k in ("T317e", "iv", "combo"):
    e = np.array([r[k][0] - r["truth"] for r in R]); C = float(np.mean([r[k][1] for r in R])); mse = float((e ** 2).mean())
    s[k] = dict(C=C, rms_pp=100 * np.sqrt(mse), bias_pp=100 * float(e.mean()), mseC=mse * C, F_len=mse * C / var,
                F_strict=mse * max(C, 1000) / .074)
for k in s:
    s[k]["mseC_ratio"] = s[k]["mseC"] / s["T317e"]["mseC"]
d = np.array([r["declared_pp"] for r in R])
s["combo"].update(declared_rms_pp=float(np.sqrt((d ** 2).mean())), declared_mean_pp=float(d.mean()), declared_max_pp=float(np.abs(d).max()),
                  go=bool(s["combo"]["mseC_ratio"] <= .70))
print(json.dumps(s, indent=1))
json.dump(dict(summary=s, rows=rows), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_combo.json"), "w"), indent=1)
