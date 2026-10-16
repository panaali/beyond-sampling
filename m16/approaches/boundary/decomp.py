import sys
import numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16"); sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/boundary")
from common import Wi, Wh, Wo
import zones2 as Z2
M = (Wi, Wh, Wo)
rng = np.random.default_rng(5)
gA = float(sys.argv[1])
G = Z2.geom_A(M, 20000, rng, gA)
use = np.minimum(G["tau"], G["mhi"]) > G["mlo"]
i = np.nonzero(use)[0]
# exact part only: monkeypatch rng-free by computing with remainder and without
import types
est, p, f = Z2.zone_line_mass(M, G["y"][i], G["d"][i], G["tau"][i], G["tn"][i], G["tp"][i], G["mlo"][i], G["mhi"][i], rng)
w = G["w"]
# exact full-zone integral for comparison
import zones as Z
ex, pc = Z.walk_wrong(M, G["y"][i], G["d"][i], G["tau"][i], G["tn"][i], G["tp"][i], G["mlo"][i], G["mhi"][i])
n = len(G["y"])
def full(v): o = np.zeros(n); o[i] = v; return w * o
E, X = full(est), full(ex)
print("est mean", E.mean(), "exact-zone mean", X.mean())
print("var est", E.var(), "var exact zone", X.var(), "var(est-exact)", (E - X).var())
yb = G["ybar"]
for lo, hi in [(-9, 0), (0, 0.3), (0.3, 0.7), (0.7, 9)]:
    m = (yb > lo) & (yb <= hi)
    print(f"ybar ({lo},{hi}] P {m.mean():.3f} mean X {X[m].mean():.4f} E[X^2] contrib {(X[m]**2).sum()/n:.5f} E[(E-X)^2] contrib {((E-X)[m]**2).sum()/n:.5f}")
print("quantiles of X/w", np.quantile(X / w, [.5, .9, .99, .999]))
# relative to zone mass
from scipy.special import ndtr
zm = 2 * (ndtr(G["tau"]) - 0.5)
r = X / w / np.maximum(zm, 1e-12)
print("frac of zone wrong quantiles", np.quantile(r[use], [.5, .9, .99]))
