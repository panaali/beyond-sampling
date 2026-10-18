"""T287 summary: per-group RMS error, F at C_eff = max(C, 1k FPE) (Directive 8 amendment 2) for each E1 design, and the
frozen-gate ideal F for D1 at E1's cost, plus exact-first-steps diagnostics. -> summary.json"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
D = json.load(open(os.path.join(HERE, "cheap.json"))); TR = json.load(open(os.path.join(ROOT, "results", "leaderboard_truths.json")))
GR = ["orig", "seed", "s0.0025", "s0.005", "s0.01", "s0.02"]
grp = lambda n: "orig" if n == "orig" else "seed" if n.startswith("seed") else n.rsplit("_", 1)[0]
out = {}
for d in ("E1_D0", "E1_D1", "E1_D2", "E1_D3"):
    C = D["orig"][d]["cost_fpe"]; Ce = max(C, 1000); row = {}
    for g in GR:
        ns = [n for n in D if grp(n) == g]
        e2 = np.mean([(D[n][d]["est"] - TR[n]["acc"]) ** 2 for n in ns]); v = np.mean([TR[n]["acc"] * (1 - TR[n]["acc"]) for n in ns])
        row[g] = dict(rms_pp=100 * float(np.sqrt(e2)), bias_pp=100 * float(np.mean([D[n][d]["est"] - TR[n]["acc"] for n in ns])), F=float(e2 * Ce / v))
    out[d] = dict(cost_fpe=C, groups=row, go=all(row[g]["F"] <= 10 for g in GR))
    print(f"{d} C={C:6d}  " + "  ".join(f"{g} rms {row[g]['rms_pp']:.2f} F {row[g]['F']:.3g}" for g in GR) + f"  go={out[d]['go']}")
ideal = {}
for key in D["orig"]["ideal_D1"]:
    row = {}
    for g in GR:
        ns = [n for n in D if grp(n) == g]
        e2 = np.mean([D[n]["ideal_D1"][key] ** 2 for n in ns]); v = np.mean([TR[n]["acc"] * (1 - TR[n]["acc"]) for n in ns])
        row[g] = dict(rms_pp=100 * float(np.sqrt(e2)), F_at_7920=float(e2 * 7920 / v))
    ideal[key] = row
    print(f"ideal D1 {key}: " + " ".join(f"{row[g]['rms_pp']:.2f}" for g in GR) + " | F@7920 " + " ".join(f"{row[g]['F_at_7920']:.3g}" for g in GR))
out["ideal_D1"] = ideal
json.dump(out, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
