"""Small cases first: M_2,2, M_4,3 and M_16,T. Deterministic co-area estimate vs truth vs sampling at equal cost.
usage: python small.py"""
import sys, time, json
import numpy as np
import core as C
from est import est_faces

res = {}
for (H, T) in [(2, 2), (4, 3), (16, 3), (16, 4), (16, 5), (16, 6), (16, 8)]:
    M = C.zoo(H, T)
    t0 = time.time()
    p, se = C.true_acc(M, n=2_000_000)
    row = {"truth": p, "truth_se": se}
    for n in (64, 256, 1024):
        n23 = n if T >= 3 else 0
        a, cost, out = est_faces(M, n23, n)
        row[n] = (a, cost)
        samp = np.sqrt(p * (1 - p) / cost) if cost > 0 else np.nan
        print(f"M_{H},{T} n={n:5d} est={a:.5f} truth={p:.5f}+-{se:.5f} err={1e2*(a-p):+.3f}pp cost={cost:8.0f} "
              f"sampling_sd@cost={1e2*samp:.3f}pp ratio={(a-p)/samp:+.2f}", flush=True)
    res[f"{H}_{T}"] = row
json.dump(res, open("small.json", "w"), indent=1, default=float)
