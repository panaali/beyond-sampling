"""D2 counterfactual: iid Neyman gain if a region's faces were integrated exactly (zero variance; cost kept)."""
import numpy as np
import core as C
def strata(kind, y, u):
    k = (y > 0).sum(1)
    if kind == "23":
        return np.where(u > 0, 2, np.where(k == 0, 0, 1))      # 0: all neg, 1: one positive (o1), 2: >=2 positives
    return np.where(u > 0, 2, 0)
print("model: gain[Neyman over strata] | zero-var all-neg (k=0) | zero-var <=1 positive | share of sqrt-var sum from <=1 pos")
for T in (4, 5):
  for s in range(5):
    D = np.load(f"../../../data/minimal_pair_d2_T{T}_s{s}.npz"); _, Z23, Z12 = C.tables(T)
    G = []; err = 0
    for kind, Z in (("23", Z23), ("12", Z12)):
        F = D[f"{kind}_F"]; c = D[f"{kind}_cost"]; st = strata(kind, D[f"{kind}_y"], D[f"{kind}_u"]); err += Z * F.mean()
        for h in (0, 1, 2):
            w = st == h
            if w.any(): G.append((h, Z * w.mean(), F[w].std(), c[w].mean()))
    p = 1 - err; v = lambda keep: sum(W * sd * np.sqrt(cc) for h, W, sd, cc in G if h in keep) * sum(W * np.sqrt(cc) * 0 + W * sd / np.sqrt(cc) * 0 for h, W, sd, cc in G) if False else None
    def var(zero):   # Neyman allocation over all strata, cost of every stratum still paid at its share
        return sum(W * (0 if h in zero else sd) * np.sqrt(cc) for h, W, sd, cc in G) ** 2
    full, z0, z01 = var(()), var((0,)), var((0, 1))
    share = sum(W * sd * np.sqrt(cc) for h, W, sd, cc in G if h < 2) / np.sqrt(full)
    print(f"T{T} s{s}: {p*(1-p)/full:6.1f}x | {p*(1-p)/z0:6.1f}x | {p*(1-p)/z01:6.1f}x | {share:.2f}")
