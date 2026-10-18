"""T297 scoring: TB variants vs the 1e7-sample MC of R0 (integration error) and vs the DEV2 1e8 truths.
usage: python tb_eval.py -> results/tb_eval.json"""
import json
import os

import numpy as np

import r0
import tb

mc = json.load(open(os.path.join(r0.HERE, "results", "mc.json")))
tr = r0.truths()
ms = dict(r0.dev2()); ms["target"] = r0.target()
VARIANTS = [(1, 1, 1), (1, 2, 1), (2, 1, 1), (2, 2, 1), (2, 2, 2), (3, 2, 2), (4, 3, 2)]
out = {}
for v in VARIANTS:
    rows = {}
    for nm in [k for k in mc if k in ms]:
        a, fl = tb.est(ms[nm], *v)
        t = tr["truths"][nm]["acc"] if nm in tr["truths"] else (0.9531174789 if nm == "target" else None)
        rows[nm] = dict(est=a, fpe=fl / r0.FPE, int_err_pp=100 * (a - mc[nm]["acc_R0"]),
                        tot_err_pp=None if t is None else 100 * (a - t))
    dev = [r for k, r in rows.items() if k != "target"]
    s = dict(variant=v, fpe=float(np.mean([r["fpe"] for r in dev])),
             int_rms_pp=float(np.sqrt(np.mean([r["int_err_pp"] ** 2 for r in dev]))),
             int_mean_pp=float(np.mean([r["int_err_pp"] for r in dev])),
             tot_rms_pp=float(np.sqrt(np.mean([r["tot_err_pp"] ** 2 for r in dev if r["tot_err_pp"] is not None]))) if dev[0]["tot_err_pp"] is not None else None,
             target_int_err_pp=rows.get("target", {}).get("int_err_pp"), rows=rows)
    out[str(v)] = s
    print(v, "FPE %.0f  integration RMS %.3f pp (mean %+.3f)  total RMS %s  target int %+.3f" % (
        s["fpe"], s["int_rms_pp"], s["int_mean_pp"], s["tot_rms_pp"] and round(s["tot_rms_pp"], 3), s["target_int_err_pp"] or 0), flush=True)
json.dump(out, open(os.path.join(r0.HERE, "results", "tb_eval.json"), "w"), indent=1)
