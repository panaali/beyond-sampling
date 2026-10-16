"""Validate wtool.walk against (1) boundary/zones.walk_wrong and (2) brute-force indicator on dense grids."""
import sys, time
import numpy as np
from scipy.special import ndtr
sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/boundary")
import wtool as W
import zones as Z
M = W.get_model("orig")
rng = np.random.default_rng(1)
y, a, b, p1, v, o1, o2 = W.surface(400, rng)
tau = W.SQ2 * np.minimum(o1 - v, v - o2)
t0 = time.time(); R = W.walk(M, y, a, b, tau); print("walk s", time.time() - t0, "mean pieces", R["pieces"].mean())
n = len(y); d = np.zeros_like(y); d[np.arange(n), a] = 1 / W.SQ2; d[np.arange(n), b] = -1 / W.SQ2
mz, _ = Z.walk_wrong(M, y, d, tau, b, a, np.zeros(n), np.full(n, np.inf))
print("max |mass - zones.walk_wrong|", np.abs(mz - R["mass"]).max())
# dense grid brute force on 40 lines
G = 20001; err_m = []; err_w = []
for i in range(40):
    t = np.linspace(-tau[i], tau[i], G)
    X = y[i] + t[:, None] * d[i]
    k = W.forward(M, X).argmax(1)
    wrong = np.where(t > 0, k != a[i], k != b[i])
    dens = np.exp(-t ** 2 / 2) / np.sqrt(2 * np.pi); dt = t[1] - t[0]
    err_m.append(((wrong * dens).sum() * dt) - R["mass"][i])
    # adjacent bands
    pos = t > 0; jp = np.argmax(~wrong[pos]) if (~wrong[pos]).any() else pos.sum()
    wp = t[pos][jp - 1] if jp > 0 else 0.0
    neg = t[::-1] < 0; wneg = wrong[::-1][neg]; jn = np.argmax(~wneg) if (~wneg).any() else neg.sum()
    wn = -t[::-1][neg][jn - 1] if jn > 0 else 0.0
    err_w.append(max(abs(wp - R["wp"][i]), abs(wn - R["wn"][i])))
print("grid: max |mass err|", np.max(np.abs(err_m)), "max |band err|", np.max(err_w), "grid dt", dt)
print("frac lines with nonzero band", (np.abs(R["delta"]) > 0).mean(), "mean mass", R["mass"].mean(), "madj share", R["madj"].sum() / R["mass"].sum())
