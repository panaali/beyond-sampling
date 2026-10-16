import sys
import numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16"); sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/boundary")
from common import Wi, Wh, Wo
from bands import line_B, band, point_out
from scipy.special import ndtr
M = (Wi, Wh, Wo)
x = np.random.default_rng(3).standard_normal((40000, 10))
LB = line_B(x)
m, sr, sl, p, first = band(M, LB["x0"], LB["d"], LB["Tp"], LB["Tn"], LB["pos"], LB["neg"])
inb = (LB["t0"] <= sr) & (LB["t0"] >= -sl)
idx = np.nonzero(inb)[0][:8]
for i in idx:
    ts = np.linspace(-min(LB["Tn"][i], .3), .3, 60001)
    xs = LB["x0"][i] + ts[:, None] * LB["d"][i]
    o = point_out(M, xs)
    mx = np.argmax(xs, 1)
    bad = o == mx
    # component containing 0
    j0 = np.argmin(np.abs(ts))
    print(i, "t0", LB["t0"][i].round(4), "sr", sr[i].round(4), "sl", sl[i].round(4), "mass", m[i],
          "grid: bad at 0", bad[j0], "approx right ext", ts[j0:][~bad[j0:]][:1], "left", ts[:j0][~bad[:j0]][-1:])
    a, b = sr[i], sl[i]
    print("   mass formula", ndtr(a) - ndtr(-b))
