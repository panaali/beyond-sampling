"""T206 summary of anatomy/*.json|npz -> anatomy_summary.json. Oracle skip budget: sampling SE at equal cost
(T196: estimator 2*ops FLOPs, c_s(H,T) per input); sort leaves by exact error mass ascending and skip the smallest ones
while their summed error stays <= SE (bias <= 1 SE). Oracle = uses the exact per-leaf error, so it is an upper bound on
what a certified per-cell bound could skip."""
import json, os, sys, glob
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import numpy as np
from leaderboard import flops
c_s = lambda H, T: flops(H, T) + T * H + 3 * T - 2
out = {}
for f in sorted(glob.glob(os.path.join(HERE, "anatomy", "*.json"))):
    k = os.path.basename(f)[:-5]; r = json.load(open(f)); z = np.load(f[:-5] + ".npz")
    H, T = int(k.split("_")[0][1:]), int(k.split("_")[1][1:]); p = r["acc"]
    se = (p * (1 - p) / (2 * r["ops"] / c_s(H, T))) ** .5
    err, mo, kind = z["err"], z["mass_ops"], z["kind"]; o = np.argsort(err); ce = np.cumsum(err[o])
    n = int(np.searchsorted(ce, se, side="right"))
    skip = dict(se=se, cells=n, cells_frac=n / len(err), err_skipped=float(ce[n - 1]) if n else 0.0,
                mass_ops_skipped_frac=float(mo[o[:n]].sum() / r["mass_ops"]), ops_after=float(r["ops"] - mo[o[:n]].sum()),
                largest_skipped_err=float(err[o[n - 1]]) if n else 0.0)
    nz = err > 0; mixed = kind == 1
    out[k] = dict(r, skip_at_1se=skip, nonzero_err_cells=int(nz.sum()), err_cells_frac=float(nz.mean()),
                  err_quantiles={q: float(np.quantile(err[nz], q)) for q in (0.1, 0.5, 0.9, 0.99)},
                  mass_ops_per_errcell=float(mo[nz].mean()), simplices_per_mixed=float(z["simplices"][mixed].mean()) if mixed.any() else 0)
    K = r["kinds"]
    print(f"{k}: ops {r['ops']:.3g} mass {100*r['mass_frac']:.1f}% leaves {r['leaves']} | correct {K['correct']['cells']} "
          f"mixed {K['mixed']['cells']} wrong {K['wrong']['cells']} (wrong err {K['wrong']['err_mass']:.2e}, mixed err {K['mixed']['err_mass']:.2e}) "
          f"| simp/massleaf {r['simplices_per_massleaf']:.1f} ops/simp {r['mass_ops_per_simplex']:.0f} | err<1e-8/7/6 frac "
          + "/".join(f"{r['error_below'][t]['err_frac']:.2e}" for t in ("1e-08", "1e-07", "1e-06"))
          + " massops " + "/".join(f"{100*r['error_below'][t]['mass_ops_frac']:.0f}%" for t in ("1e-08", "1e-07", "1e-06"))
          + f" | SE {se:.2e}: skip {n} cells ({100*skip['cells_frac']:.0f}%), {100*skip['mass_ops_skipped_frac']:.0f}% mass ops, max err {skip['largest_skipped_err']:.1e}")
json.dump(out, open(os.path.join(HERE, "anatomy_summary.json"), "w"), indent=1)
