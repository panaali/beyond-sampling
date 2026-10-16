"""Per-stratum variance/cost of swap estimators (original model) -> Neyman-optimal efficiency vs plain."""
import sys, itertools, json
import numpy as np
sys.path.insert(0, "m16"); sys.path.insert(0, "m16/approaches/tie_lines")
from common import Wi, Wh, Wo
from tielines import correct
from stratified import stratum, N_STRATA
M = (Wi, Wh, Wo)
P = np.array(json.load(open("m16/stratified_design.json"))["P"])
N = 400000
x = np.random.default_rng(4).standard_normal((N, 10))
o = np.argsort(-x, 1); xs = np.take_along_axis(x, o, 1); r = np.arange(N)
G = {}
for p in itertools.permutations(range(3)):
    y = x.copy()
    for k in range(3): y[r, o[:, k]] = x[r, o[:, p[k]]]
    G[p] = correct(M, y).astype(float)
g = G[(0, 1, 2)]; s23 = G[(0, 2, 1)]; s12 = G[(1, 0, 2)]
S3 = np.mean([G[p] for p in G], 0)
g23 = (xs[:, 1] - xs[:, 2]) / np.maximum(np.abs(xs[:, 1]), 1e-12); g12 = (xs[:, 0] - xs[:, 1]) / np.abs(xs[:, 0])
h = stratum(x)
ests = {"plain": (g, 1), "swapmin": (np.where(g23 < g12, (g + s23) / 2, (g + s12) / 2), 2),
        "swap2": ((g + s23 + s12) / 3, 3), "S3": (S3, 6), "swap23": ((g + s23) / 2, 2)}
vg = g.var()
tab = {}
for k, (e, c) in ests.items():
    tab[k] = [e[h == s].std() for s in range(N_STRATA)]
    print(k, " ".join("%.3f" % v for v in tab[k]))
# best method per stratum by sigma*sqrt(cost)
best = []
tot = 0
for s in range(N_STRATA):
    sc = {k: tab[k][s] * np.sqrt(ests[k][1]) for k in ests}
    b = min(sc, key=sc.get); best.append(b); tot += P[s] * sc[b]
print("best per stratum", best)
print("strat+plain eff %.2f" % (vg / sum(P[s] * tab["plain"][s] for s in range(N_STRATA)) ** 2))
print("strat+best eff %.2f" % (vg / tot ** 2))
