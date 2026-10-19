"""T405 post hoc diagnostic (not pre-registered): central FD of the frozen T5 *value* along a direction at several
steps, vs T5's analytic slope and the truth. Separates a wrong pointwise derivative (aliasing / near-tangent endpoints)
from a wrong estimand slope. usage: python diag_fd.py <model> <t0> <dir> <step,...> -> results/diag_fd.log (append)"""
import json
import sys

import numpy as np

SH = [(16,), (16, 16), (10, 16)]


def unflat(v):
    out, i = [], 0
    for s in SH:
        n = int(np.prod(s)); out.append(v[i:i + n].reshape(s)); i += n
    return tuple(out)


import frozen
import make_cases

nm, t0, i = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]); steps = [float(s) for s in sys.argv[4].split(",")]
C = json.load(open(make_cases.path(nm, t0))); w = np.array(C["w"]); u = np.array(C["dirs"][i])
T = json.load(open(f"results/truth_{nm}_{t0}.json")); P = T["P"]
for h in steps:
    ep = frozen.transversal.est_grad(unflat(w + h * u), t0, frozen.KT, frozen.V2SPEC, want_grad=False)["est"]
    em = frozen.transversal.est_grad(unflat(w - h * u), t0, frozen.KT, frozen.V2SPEC, want_grad=False)["est"]
    print(nm, t0, "dir", i, "step %.2e" % h, "FD(T5 value)/P %.1f" % ((ep - em) / (2 * h) / P), "analytic T5/P %.1f" % (C["t5"]["pred"][i] / P),
          "truth FD_sel/P %.1f" % (T["FD_sel"][i] / P), flush=True)
