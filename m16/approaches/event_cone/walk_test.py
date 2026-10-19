"""Side check (DRI): does the truth worker's conditional walk (truth_t398/truth.py, T362's E1 path) see the MC hits?
For each hit x (pick = p3, g23 >= t0 at x itself), compute Z(x) = P(pick = p3, v2 - x_p3 > t0 | the other 9 values)
with the SAME code path (rare.walk3 along x_p3 from v2 down to v4). Every hit lies on its own line, so Z(x) > 0 is
required; Z = 0 on a hit means the walk misses event mass (a bias in the truth). Also reports the forward check.
usage: python walk_test.py hits.npz [t0]"""
import os
import sys

import numpy as np
from scipy.special import ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "truth_t398", "src")); sys.path.insert(0, os.path.join(HERE, "truth_t398"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import rare  # noqa: E402
from truth import EPS, orig  # noqa: E402
import common  # noqa: E402

x = np.load(sys.argv[1])["x"]; t0 = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5
model = orig()
o = np.argsort(-x, 1); r = np.arange(len(x)); v = x[r[:, None], o]
v2, v4 = v[:, 1], v[:, 3]; p2, p3 = o[:, 1], o[:, 2]
pick = (common.run(x)[:, -1] @ common.Wo.T).argmax(1)
s = v2 - EPS; x0 = x.copy(); x0[r, p3] = s
d = np.zeros_like(x0); d[r, p3] = -1.0
pk3, pkc, f, it, nc = rare.walk3(model, x0, d, p2, p3, s - v4)
mass = np.zeros(len(x)); covered = np.zeros(len(x), bool)
for rows, a, b in pk3:
    # does the hit's own offset (s - x_p3) lie in a pick=p3 interval?
    off = s[rows] - x[rows, p3[rows]]
    covered[rows[(a <= off) & (off <= b)]] = True
    a2 = np.maximum(a, t0 - EPS); m = b > a2
    np.add.at(mass, rows[m], ndtr(s[rows[m]] - a2[m]) - ndtr(s[rows[m]] - b[m]))
Z = mass / (ndtr(v2) - ndtr(v4))
for i in range(len(x)):
    print(f"hit {i}: tri ({o[i,0]},{o[i,1]},{o[i,2]}) g23 {v2[i]-v[i,2]:.3f} pick==p3 {pick[i]==p3[i]} "
          f"walk covers own point {covered[i]} Z {Z[i]:.3g}")
print("hits with Z == 0:", int((Z == 0).sum()), "of", len(x), "| own point not covered:", int((~covered).sum()))
