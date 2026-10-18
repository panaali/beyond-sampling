"""T297 addendum (DRI, 1.8k FPE target): cheaper TB variants down to ~1k FPE. usage: python tb_cheap.py -> results/tb_cheap.json"""
import json
import os

import numpy as np

import r0
import tb

mc = json.load(open(os.path.join(r0.HERE, "results", "mc.json"))); ms = dict(r0.dev2())
V = [dict(n23=1, newton=1, nblind=0, use12=False), dict(n23=1, newton=1, nblind=1, use12=False),
     dict(n23=1, newton=1, nblind=0, use12=True), dict(n23=1, newton=1, nblind=1, use12=True)]
out = []
for v in V:
    errs, fpe = [], []
    for nm in [k for k in mc if k in ms]:
        a, fl = tb.est(ms[nm], **v); errs.append(100 * (a - mc[nm]["acc_R0"])); fpe.append(fl / r0.FPE)
    r = dict(variant=v, fpe=float(np.mean(fpe)), int_rms_pp=float(np.sqrt(np.mean(np.square(errs)))), int_mean_pp=float(np.mean(errs)))
    out.append(r); print(r, flush=True)
json.dump(out, open(os.path.join(r0.HERE, "results", "tb_cheap.json"), "w"), indent=1)
