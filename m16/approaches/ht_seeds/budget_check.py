"""Does the collapse point depend on the budget? Re-measure the Sobol var/MSE gain at n faces per kind (default
4096, 4x the scan) for chosen cells, reusing the independent truths in scan_seeds.json.
usage: OMP_NUM_THREADS=1 python budget_check.py --n 4096 --R 16 --cells 16:4 16:5 16:6 32:5 32:6"""
import argparse, json, os
import numpy as np
from scipy.stats import chi2
import core as C
from est import est_faces

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--cells", nargs="+", required=True); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
ap.add_argument("--n", type=int, default=4096); ap.add_argument("--R", type=int, default=16)
a = ap.parse_args()
D = json.load(open(os.path.join(HERE, "scan_seeds.json")))
OUT = os.path.join(HERE, f"budget_n{a.n}.json")
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
for spec in a.cells:
    H, T = map(int, spec.split(":"))
    for sd in a.seeds:
        key = f"{H}:{T}:{sd}"
        if key not in out:
            M = C.zoo(H, T, seed=sd); e, cs = [], []
            for r in range(a.R):
                acc, c, _ = est_faces(M, a.n, a.n, rule_seed=700 + r); e.append(float(acc)); cs.append(float(c))
            out[key] = dict(est=e, cost=cs, n=a.n); json.dump(out, open(OUT, "w"), indent=1)
        tr = D[key]["truth"]; pt, se = tr["acc"], tr["se"]
        e = np.array(out[key]["est"]); c = np.mean(out[key]["cost"]); v = e.var(ddof=1); R = len(e)
        g = pt * (1 - pt) / c / v
        g1 = pt * (1 - pt) / np.mean(D[key]["sobol"]["cost"]) / np.var(D[key]["sobol"]["est"], ddof=1)
        mse = max(np.mean((e - pt) ** 2) - se ** 2, v / R)
        print(f"H={H:2d} T={T:2d} seed={sd}: n={a.n} var-gain={g:.1f} [{g * chi2.ppf(.025, R - 1) / (R - 1):.1f}, "
              f"{g * chi2.ppf(.975, R - 1) / (R - 1):.1f}] mse-gain={pt * (1 - pt) / c / mse:.1f} "
              f"(truth-resolved: {se < 0.5 * np.sqrt(v)}) | n=1024 var-gain={g1:.1f} | cost {c:.0f}", flush=True)
