"""T297 reference (DIAGNOSTIC sampling): acc_R0 from 1e7 skeletons of the exact law, with R0's error split by region:
blind (v2 <= 0), band23 (g23 < .05), band12 (g12 < .05), other. usage: python mc.py [n] -> results/mc.json"""
import json
import os
import sys
import time

import numpy as np

import r0

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000
ms = dict(r0.dev2()); names = [list(ms)[i] for i in r0.DEV_IDX]
fn = os.path.join(r0.HERE, "results", "mc.json"); os.makedirs(os.path.dirname(fn), exist_ok=True)
out = json.load(open(fn)) if os.path.exists(fn) else {}
models = [(nm, ms[nm]) for nm in names] + [("target", r0.target())]
for nm, model in models:
    if nm in out:
        continue
    t0 = time.time(); rng = np.random.default_rng(297_000 + len(out)); ok = 0; err = np.zeros(4)
    for s in range(0, n, 250_000):
        tri, v1, v2, v3 = r0.sample_skeleton(250_000, rng)
        c = r0.r0_correct(model, tri, v1, v2, v3)
        ok += c.sum()
        reg = np.where(v2 <= 0, 0, np.where(v2 - v3 < .05, 1, np.where(v1 - v2 < .05, 2, 3)))
        err += np.bincount(reg[~c], minlength=4)
    out[nm] = dict(acc_R0=ok / n, se=float(np.sqrt(ok / n * (1 - ok / n) / n)),
                   err_by_region=dict(zip(("blind", "band23", "band12", "other"), (err / n).tolist())), sec=time.time() - t0)
    print(nm, json.dumps(out[nm]), flush=True)
    json.dump(out, open(fn, "w"), indent=1)
