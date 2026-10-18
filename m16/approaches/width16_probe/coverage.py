"""T221: mass coverage curve of exact prefix regions. Region mass = sum of the exact masses of the T217 pieces
(eventword/<model>_k{k}.npz) with the same gate word. N_delta = number of heaviest regions covering 1 - delta.
-> coverage.json"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6); out = {}
for m in ("target", "zoo16_4", "zoo16_5"):
    prev = None
    for k in (2, 3, 4):
        z = np.load(os.path.join(HERE, "eventword", f"{m}_k{k}.npz"))
        w = np.bincount(z["gate"], weights=z["mass"]); w = np.sort(w)[::-1]; tot = w.sum(); c = np.cumsum(w) / tot
        N = {f"{d:.0e}": int(np.searchsorted(c, 1 - d) + 1) for d in D}
        r = dict(regions=len(w), mass_sum_minus_1=float(tot - 1), N=N, below_1e12_mass=float(w[w < 1e-12].sum()),
                 below_1e12_count=int((w < 1e-12).sum()), min_mass=float(w[-1]), median_mass=float(np.median(w)),
                 growth=None if prev is None else {d: N[d] / prev["N"][d] for d in N}, regions_growth=None if prev is None else len(w) / prev["regions"])
        out[f"{m}_k{k}"] = r; prev = r
        print(m, k, len(w), N, "growth", r["growth"] and {d: round(v, 1) for d, v in r["growth"].items()}, "reg x%s" % (r["regions_growth"] and round(r["regions_growth"], 1)),
              "mass<1e-12 %.1e (%d regions)" % (r["below_1e12_mass"], r["below_1e12_count"]))
json.dump(out, open(os.path.join(HERE, "coverage.json"), "w"), indent=1)
