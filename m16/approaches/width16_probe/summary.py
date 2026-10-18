"""T194 summary -> summary.json and table rows. Ratio = (recipe - exact)^2 / (p(1-p)/N), N = recipe ops / fpe_flops(H,T)
(estT_fix convention). z = (exact - MC1e7, rng 2026) / se; z2 vs 2e7 MC rng 7 (mc2.py). (8,4) recipe/exact numbers from estT_fix/rescore (same code)."""
import json, os, sys, glob
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", ".."))
from leaderboard import fpe_flops
mc = json.load(open(os.path.join(HERE, "mc.json"))); out = {}
mc2 = json.load(open(os.path.join(HERE, "mc2.json")))


def res(H, T, n, mode):
    fn = os.path.join(HERE, "res", f"H{H}_T{T}_{n}_{mode}.json")
    if os.path.exists(fn):
        return json.load(open(fn))
    fn = os.path.join(HERE, "..", "estT_fix", "rescore", n + ".json")
    if (H, T) == (8, 4) and os.path.exists(fn):
        return json.load(open(fn))[mode]


for k, v in sorted(mc.items()):
    H, T = int(k.split("_")[0][1:]), int(k.split("_")[1][1:]); n = k.split("_", 2)[2]
    rc, ex = res(H, T, n, "recipe108"), res(H, T, n, "exact")
    row = dict(v); row.update(H=H, T=T, name=n)
    if ex and ex.get("acc") is not None:
        p = ex["acc"]; se = (v["mc1e7"] * (1 - v["mc1e7"]) / 1e7) ** .5
        row.update(exact=p, z_exact_mc=(p - v["mc1e7"]) / se, exact_ops=ex["ops"], exact_cells=ex["cells"],
                   exact_leafsum_minus_1=ex.get("leafsum_minus_1"), regions_exact=ex.get("regions_exact"))
        if k in mc2:
            row.update(mc2_2e7=mc2[k], z_exact_mc2=(p - mc2[k]) / (mc2[k] * (1 - mc2[k]) / 2e7) ** .5)
    if rc:
        row.update(recipe_stopped=rc.get("stopped"), ops=rc["ops"], cells=rc["cells"], recipe_cpu=rc.get("cpu", rc.get("sec")))
        if rc.get("acc") is not None and "exact" in row:
            row.update(est=rc["acc"], err_pp=100 * (rc["acc"] - p),
                       ratio=(rc["acc"] - p) ** 2 / (p * (1 - p) / (rc["ops"] / fpe_flops(H, T))))
        reg = row.get("regions_exact") or v["regions_lb_2e6"]
        row.update(cells_per_region=rc["cells"] / reg, cells_per_region_lb2e5=rc["cells"] / v["regions_lb_2e5"])
    out[k] = row
    f = lambda x, s: (s % x) if isinstance(x, (int, float)) else str(x)
    print(k, "acc", f(row.get("exact"), "%.8f"), "z", f(row.get("z_exact_mc"), "%+.2f"), f(row.get("z_exact_mc2"), "%+.2f"), "err", f(row.get("err_pp"), "%+.5f"),
          "ratio", f(row.get("ratio"), "%.2g"), "ops", f(row.get("ops"), "%.3g"), "cells", row.get("cells"),
          "reg", row.get("regions_exact"), v["regions_lb_2e6"], "c/r", f(row.get("cells_per_region"), "%.2f"),
          "leaf", f(row.get("exact_leafsum_minus_1"), "%.1e"))
json.dump(out, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
