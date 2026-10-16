"""Ideal Neyman gain from stratifying the face rule on the tie value u (model-free strata; allocation from the
model's own faces). Prints the MSE gain over sampling at equal compute for proportional vs Neyman allocation.
usage: python diag_neyman.py model_name n nbins"""
import sys, json
import numpy as np
import core as C
sys.path.insert(0, "/home/user/clean-neurons/m16")
from arc_eval import Wi0, Wh0, Wo0, noisy_models
name = sys.argv[1]; n = int(sys.argv[2]); nb = int(sys.argv[3])
if name == "orig": M = (Wi0, Wh0, Wo0)
elif name.startswith("seed"): M = C.zoo(16, 10, seed=int(name[4:]))
else:
    sig, k = name[1:].split("_"); M = noisy_models(6, float(sig), seed=0)[int(k)]
T = 10
_, Z23, Z12 = C.tables(T)
strata = []  # (weight W_h in accuracy units, mean, sd, cost)
for kind, Z in (("23", Z23), ("12", Z12)):
    U = C.sobol(T, n, 5)
    y, a, b, tau = C.face_points(T, kind, U)
    F, cost, _ = C.walk_mass(M, y, a, b, tau, kind)
    edges = np.r_[0, np.geomspace(1e-3, 1, nb)[:-1] * 0 + np.linspace(0, 1, nb + 1)[1:-1], 1] if False else np.linspace(0, 1, nb + 1)
    # finer bins at low u (blind side): use quantile edges in U[:,1] concentrated near 0
    edges = np.r_[0, 0.002, 0.01, 0.03, 0.07, np.linspace(0.15, 1, nb - 4)]
    hb = np.clip(np.searchsorted(edges, U[:, 1], side="right") - 1, 0, len(edges) - 2)
    for h in range(len(edges) - 1):
        m = hb == h
        if m.sum() < 5: continue
        P = edges[h + 1] - edges[h]
        strata.append((Z * P, F[m].mean(), F[m].std(), cost[m].mean(), kind, h, m.sum()))
S = np.array([s[:4] for s in strata])
W, mu, sd, c = S.T
est = 1 - (W * mu).sum(); p = est
avgc = None
# proportional (to W) allocation: nodes n_h ∝ W_h; cost per unit node = sum W c / sum W
prop = (W.sum()) * (W * sd ** 2).sum() * (W * c).sum() / W.sum() ** 2   # variance x cost
ney = ((W * sd * np.sqrt(c)).sum()) ** 2
samp = p * (1 - p)
print(f"{name}: est {p:.5f}  Var*cost: sampling {samp:.4g}  face-prop {prop:.4g} (gain {samp/prop:.2f})  face-Neyman-u {ney:.4g} (gain {samp/ney:.2f})")
for s in strata:
    print("   R%s bin%d n=%d W=%.4f mean=%.4f sd=%.4f cost=%.1f contrib(W sd sqrt c)=%.5f" % (s[4], s[5], s[6], s[0], s[1], s[2], s[3], s[0]*s[2]*np.sqrt(s[3])))
