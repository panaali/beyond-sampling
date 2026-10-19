"""D2: oracle Neyman gain (iid, known stratum sd) for strata = sign, config, config x sign, + u-bins. Upper bounds."""
import numpy as np
import core as C
def sign_st(kind, y, u):
    k = (y > 0).sum(1)
    return np.where(u > 0, 2, np.where(k == 0, 0, 1)) if kind == "23" else np.where(u > 0, 2, 0)
for T in (4, 5):
  for s in range(5):
    D = np.load(f"../../../data/minimal_pair_d2_T{T}_s{s}.npz"); _, Z23, Z12 = C.tables(T)
    res = {}
    for name in ("kind", "sign", "config", "config x sign", "config x sign x u4"):
        S = 0; err = 0
        for kind, Z in (("23", Z23), ("12", Z12)):
            F = D[f"{kind}_F"]; c = D[f"{kind}_cost"]; err += Z * F.mean()
            key = np.zeros(len(F), int)
            if "sign" in name: key = key * 3 + sign_st(kind, D[f"{kind}_y"], D[f"{kind}_u"])
            if "config" in name: key = key * 100 + D[f"{kind}_ci"]
            if "u4" in name: key = key * 4 + np.digitize(D[f"{kind}_u"], np.quantile(D[f"{kind}_u"], [.25, .5, .75]))
            for k in np.unique(key):
                w = key == k; S += Z * w.mean() * F[w].std() * np.sqrt(c[w].mean())
        p = 1 - err; res[name] = p * (1 - p) / S ** 2
    print(f"T{T} s{s}: " + "  ".join(f"{k} {v:5.1f}x" for k, v in res.items()))
