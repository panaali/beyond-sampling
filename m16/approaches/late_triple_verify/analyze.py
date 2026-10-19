"""T418: projections from lbab run curves (pre-registered rule: log-log fit of charge vs boxes over the last decade,
extrapolated to 1e-10; stall = < 2x decrease over that decade). usage: python analyze.py RUN.json [...]"""
import json
import sys

import numpy as np

for p in sys.argv[1:]:
    R = json.load(open(p)); c = R["curve"]
    e = np.array([r["evals"] + (r.get("abc_dom", 0) if "abc" in R["stages"] else 0) for r in c], float)
    e0 = np.array([r["evals"] for r in c], float); f = c[-1]
    print("%s tri=%s order=%s split=%s boxes=%d pruned=%.1f%% cpu=%.0fs" % (p, tuple(R["tri"]), R.get("order", "radial"),
          R.get("split", "margin"), f["evals"], 100 * f["pruned"] / max(f["evals"], 1), f["cpu_s"]))
    for key in ["radial", "quad"]:
        for unit, x in ([("boxes", e0), ("boxes+domains", e)] if "abc" in R["stages"] else [("boxes", e0)]):
            v = np.array([r[key] for r in c]); m = x >= x[-1] / 10
            s = np.polyfit(np.log(x[m]), np.log(v[m]), 1)[0]; drop = v[m][0] / v[-1]
            proj = x[-1] * (1e-10 / v[-1]) ** (1 / s) if s < 0 else float("inf")
            print("  %-6s %-13s final %.3e  slope %.3f  drop/decade %.2f  %s  projected %s to 1e-10: %.2e" % (
                key, unit, v[-1], s, drop, "STALL" if drop < 2 else "", unit, proj))
