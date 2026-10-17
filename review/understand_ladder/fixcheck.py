"""T97: does the id()-reuse fix remove F2L2's negative errors? Re-scores every model of score_H{H}.json whose
error was < -1e-12 (any LOOK), plus the adversarial worst models, with fibre_fix (same code + keep-alive).
usage: OMP_NUM_THREADS=1 python fixcheck.py  -> fixcheck.json"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "m16", "approaches", "understand_ladder"))
import fibre_fix  # noqa: E402
import rtruth as R  # noqa: E402
from score import models  # noqa: E402

out = {}
for H in (4, 5, 6):
    J = json.load(open(os.path.join(HERE, f"score_H{H}.json")))["rows"]
    neg = sorted({n for n, r in J.items() for L in (1, 2, 3) if r[f"L{L}"]["est"] - r["truth"] < -1e-12})
    ms = models(H)
    A = json.load(open(os.path.join(HERE, f"adv_H{H}.json")))
    todo = [(n, ms[n], J[n]) for n in neg] + [("adv_worst", tuple(np.array(w) for w in A["worst"]["model"]), None)]
    for n, m, old in todo:
        tr = R.exact(m)[0] if old is None else old["truth"]
        row = {}
        for L in (1, 2, 3):
            fibre_fix.LOOK = L; o = fibre_fix.Ops(); e = fibre_fix.est_fibre2(m, o, {})
            row[f"L{L}"] = dict(old=None if old is None else old[f"L{L}"]["est"] - tr, fixed=e - tr, ops=o.n)
        out[f"H{H}_{n}"] = row
        print(f"H={H} {n:18s} " + "  ".join(f"L{L} old {row[f'L{L}']['old'] if row[f'L{L}']['old'] is not None else float('nan'):+.1e} fixed {row[f'L{L}']['fixed']:+.1e}" for L in (1, 2, 3)), flush=True)
json.dump(out, open(os.path.join(HERE, "fixcheck.json"), "w"), indent=1)
