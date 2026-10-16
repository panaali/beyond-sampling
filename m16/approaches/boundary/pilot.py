"""Pilot on the ORIGINAL model only: per-line values and costs for A and B, used to choose strata allocation."""
import sys
import numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16"); sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/boundary")
from common import Wi, Wh, Wo
import zones2 as Z2
M = (Wi, Wh, Wo)
gA, gB, n = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
rng = np.random.default_rng(42)
out = {}
for name, G in [("A", Z2.geom_A(M, n, rng, gA)), ("B", Z2.geom_B(M, n, rng, gA, gB))]:
    v, p, f = Z2.term(M, G, rng)
    a = np.argmax(G["d"] > 0, 1); b = np.argmax(G["d"] < 0, 1)
    out[name] = dict(v=v, f=f, ybar=G["ybar"], a=a, b=b)
    print(name, "mean", v.mean(), "flops", f.mean(), "s", np.sqrt(v.var() * f.mean()))
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    pair = lo * 10 + hi
    def ney(lab):
        S = 0
        for h in np.unique(lab):
            m = lab == h; P = m.mean()
            S += P * v[m].std() * np.sqrt(f[m].mean() + 1e-9)
        return S
    q = np.quantile(G["ybar"], [0.25, 0.5, 0.75]); yb = np.searchsorted(q, G["ybar"])
    print("  neyman s: pair", ney(pair), " ybar4", ney(yb), " pair x ybar4", ney(pair * 10 + yb), " hi-pos", ney(hi), "hi x ybar", ney(hi*10+yb))
np.savez(f"pilot_{gA}_{gB}.npz", **{k + "_" + kk: vv for k, d in out.items() for kk, vv in d.items()})
