"""T302: walk-length scan on the 10 T297 DEV2 models (integration error vs 1e7 MC of R0; total error vs truths)."""
import json
import os
import sys

import numpy as np

import walk
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r0_integral"))
import r0  # noqa: E402

mc = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r0_integral", "results", "mc.json")))
ms = dict(r0.dev2()); tr = r0.truths()["truths"]; names = [k for k in mc if k in ms]
out = {}
for tmax in [float(a) for a in sys.argv[1:]] or [0.25, 0.5, 1.0, 2.0, 6.0]:
    ie, te, fp, nb, bl = [], [], [], [], []
    for nm in names:
        a, f, p = walk.est(ms[nm], parts=True, tmax=tmax)
        ie.append(100 * (a - mc[nm]["acc_R0"])); te.append(100 * (a - tr[nm]["acc"])); fp.append(f / r0.FPE)
        nb.append(100 * ((1 - 11 / 1024 - p["nb"]) - sum(v for k, v in mc[nm]["err_by_region"].items() if k != "blind")))
        bl.append(100 * ((11 / 1024 - p["b"]) - mc[nm]["err_by_region"]["blind"]))
    r = dict(tmax=tmax, fpe=float(np.mean(fp)), int_rms=float(np.sqrt(np.mean(np.square(ie)))), int_mean=float(np.mean(ie)),
             tot_rms=float(np.sqrt(np.mean(np.square(te)))), err_nb_minus_mc=float(np.mean(nb)), err_blind_minus_mc=float(np.mean(bl)),
             err_nb_rms=float(np.sqrt(np.mean(np.square(nb)))), err_blind_rms=float(np.sqrt(np.mean(np.square(bl)))))
    out[str(tmax)] = r; print({k: round(v, 3) for k, v in r.items()}, flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_scan.json"), "w"), indent=1)
