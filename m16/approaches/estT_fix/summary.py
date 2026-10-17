"""T131/T164 summary -> summary.json; prints the table rows. Ratio = (est - exact)^2 / (p(1-p)/N), N = ops / FPE."""
import json, os, sys, glob
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", ".."))
from leaderboard import fpe_flops
mc = json.load(open(os.path.join(HERE, "mc_H8_T4.json"))); old = json.load(open(os.path.join(HERE, "..", "final3", "res_H8_T4.json")))
F = fpe_flops(8, 4); out = {}; agg = {}
for fn in sorted(glob.glob(os.path.join(HERE, "rescore", "*.json"))):
    n = os.path.basename(fn)[:-5]; r = json.load(open(fn)); p = r["exact"]["acc"]; rc = r["recipe108"]
    se = (mc[n] * (1 - mc[n]) / 1e7) ** .5; o = old[n]
    row = dict(exact=p, mc1e7=mc[n], z_exact_mc=(p - mc[n]) / se, est=rc["acc"], err_pp=100 * (rc["acc"] - p), ops=rc["ops"],
               ratio=(rc["acc"] - p) ** 2 / (p * (1 - p) / (rc["ops"] / F)), cells=rc["cells"], regions_lb=o["regions_lb"],
               old_est=o["est"], old_err_pp=100 * (o["est"] - p), old_ops=o["ops"],
               old_ratio=(o["est"] - p) ** 2 / (p * (1 - p) / (o["ops"] / F)), splits=r["splits"])
    for t, s in r["splits"].items():
        a = agg.setdefault(t, [0, 0, 0]); a[0] += s["n"]; a[1] += s["class_changes"]; a[2] += s["err_changes"]
    out[n] = row
    print(f"{n} z {row['z_exact_mc']:+.2f} err {row['err_pp']:+.5f}pp ops {row['ops']:.3g} ratio {row['ratio']:.3g} | old {row['old_err_pp']:+.4f}pp {row['old_ratio']:.3g} cells {row['cells']} reg {row['regions_lb']}")
out["oversplit_by_step"] = {t: dict(splits=a[0], class_change=a[1], err_change=a[2], oversplit_class=1 - a[1] / a[0],
                                    oversplit_err=1 - a[2] / a[0]) for t, a in agg.items()}
print({t: (v["splits"], round(v["oversplit_class"], 3), round(v["oversplit_err"], 3)) for t, v in out["oversplit_by_step"].items()})
json.dump(out, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
