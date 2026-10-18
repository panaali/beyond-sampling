"""T307 B2 (cost only): exact prefix-sharing count for the FIRST cell of every walked line of R1W V1 (inputs and
tangents there are model-independent). Distinct (x[:, :s+1], dx[:, :s+1]) prefixes across the 720 triples, per step."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "r1_walk")); sys.path.insert(0, os.path.join(HERE, "..", "r0_walk"))
sys.path.insert(0, os.path.join(HERE, "..", "r0_integral"))
import r0  # noqa: E402
import r1w  # noqa: E402
import walk as W  # noqa: E402

tot_sep = tot_shared = 0.0
for name, lo, hi, (v2s, g12s, g23s) in __import__("r1w2").REG[False]:
    v3s = v2s - g23s; v4s = float(r1w.gauss_v4(v3s, 1)[0][0]); delta = v3s - v4s
    m4 = float(r0.mu(np.array([v4s]))[0]); dm4 = float(r1w.dmu(v4s)[0])
    lines = [r1w.inputs(v2s + g12s, v2s, v2s, v2s - delta, m4 + dm4 * (v2s - delta - v4s), 0.0, -1.0, -1.0, -dm4)]
    if name.startswith("nb"):
        lines.append(r1w.inputs(v2s, v2s, v3s, v4s, m4, 1.0, 0.0, 0.0, 0.0))
    for x, dx in lines:
        sep = 720 * 10                                                     # step evaluations without sharing
        sh = sum(len({(tuple(np.round(x[i, :s + 1], 12)), tuple(np.round(dx[i, :s + 1], 12))) for i in range(720)})
                 for s in range(10))
        tot_sep += sep; tot_shared += sh
        print(name, "first-cell step evals: separate", sep, "shared", sh, "ratio %.2f" % (sep / sh))
print("first cells, all lines: FPE separate %.0f -> shared %.0f (x2 for value+tangent, /T per step)" % (
    2 * tot_sep / 10, 2 * tot_shared / 10))
