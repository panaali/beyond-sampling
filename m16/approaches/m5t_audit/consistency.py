"""T417: consistency check for the Q-level lo flags (Q does not depend on the tangent centre): fast lo with the
adversarial centre must be <= sum of fast hi over 2^K bisection children with the production (Wolfe) centre.
Children volumes are taken as vol / 2^K with 1e-12 slack (float midpoints), so this is a check, not a certificate.
usage: python consistency.py -> results/consistency.json"""
import json
import math
import os
import sys

import numpy as np
from flint import fmpq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "event_cone_rigour"))
sys.path.insert(0, HERE)
import m5t_fast as F  # noqa: E402

import gen  # noqa: E402
from qstrat import pieces  # noqa: E402

res = []
for s in (2, 3):
    for w in json.load(open(os.path.join(HERE, "results", f"qcheck_{s}.json")))["worst"]:
        M = np.array(w["M"]); d, mlow = w["d"], w["mlow"]
        for a in range(d):
            for b in range(d):
                if a != b and np.all(M[:, a] == 1 + M[:, b]):
                    jp2, jp3 = a, b
        vol = gen.volume(M, jp2)
        zr = fmpq(*[int(v) for v in w["zratio"].split("/")]) if "/" in w["zratio"] else fmpq(int(w["zratio"]))
        P = F.Params(d, mlow, jp3, w["t0"], w["zbins"], zr); vlo, vhi = F.q2iv(vol)
        lo, _ = F.bounds_batch(M[None], M[None], np.array([vlo]), np.array([vhi]), P, mc=np.array(w["mc"])[None])
        ub = []
        for K in (0, 6, 10):
            C = pieces(M, K); n = len(C)
            _, ch = F.bounds_batch(C, C, np.full(n, vlo / n * (1 - 1e-12)), np.full(n, vhi / n * (1 + 1e-12)), P)
            ub.append(math.fsum(ch) * (1 + 1e-12))
        res.append(dict(lo_adversarial=float(lo[0]), ub_wolfe_K0_6_10=ub, consistent=bool(lo[0] <= min(ub))))
json.dump(dict(cases=len(res), consistent=sum(r["consistent"] for r in res), rows=res),
          open(os.path.join(HERE, "results", "consistency.json"), "w"), indent=0)
print(len(res), sum(r["consistent"] for r in res))
