"""Aggregates step1a.json and cells_*.npz into analysis.log / analysis.json (scores per the pre-registration)."""
import json
import numpy as np
import plwalk as P
from cells import LEVELS

T449_W0 = {12: .262, 36: .0977}
lines_out, res = [], {"step1a": json.load(open("step1a.json"))}
say = lambda s="": (print(s), lines_out.append(s))
say("STEP 1a: rho = |P c| / |c.d| (E: decision roots, G: gate roots), 2,048 frozen nodes x 5 strata per model")
ok_h, grow = {"E": 0, "G": 0}, {"E": 0, "G": 0}
for m, r in res["step1a"].items():
    say(f"  {m}: " + " | ".join(f"{k}: n {r[k]['n']} med {r[k]['median']:.3g} p99 {r[k]['p99']:.3g} max {r[k]['max']:.3g} "
                                 f"(256 nodes {r[k]['max256']:.3g}) Hill 1/2/5% {r[k]['hill1']:.2f}/{r[k]['hill2']:.2f}/{r[k]['hill5']:.2f}"
                                 for k in "EG") + f" | max k(s*)rho {r['E']['max_k_rho']:.3g}")
    ok_h["E"] += .9 <= r["E"]["hill2"] <= 2.7; ok_h["G"] += 1.0 <= r["G"]["hill2"] <= 2.0
    for k in "EG":
        grow[k] += r[k]["max"] > r[k]["max256"]
say(f"  Hill(2%) inside widened T449 ranges: E {ok_h['E']}/5, G {ok_h['G']}/5; max grows 256 -> 2048 nodes: "
    f"E {grow['E']}/5, G {grow['G']}/5 (pre-registered: >= 4/5 each)")

say("\nSTEPS 1b / 2: per level, sum over strata of M_h x (cell statistic); 400 cells x 64 points per stratum")
W = {}
for m in P.MODELS:
    z = np.load(f"cells_{m}.npz")
    for k in LEVELS:
        W0 = sum(z[f"M_{j}"] * z[f"r0_{j}_{k}"].mean() for j in range(5))
        W1 = sum(z[f"M_{j}"] * z[f"r1_{j}_{k}"].mean() for j in range(5))
        nsm = [int(z[f"sm_{j}_{k}"].sum()) for j in range(5)]
        if sum(nsm) >= 20 and min(nsm) >= 1:
            WCA = sum(z[f"M_{j}"] * np.median(z[f"r1_{j}_{k}"][z[f"sm_{j}_{k}"]]) for j in range(5))
        else:
            WCA = np.nan
        W[(m, k)] = (W0, W1, WCA, sum(nsm) / 2000, min(nsm))
res["cells"] = {f"{m}|{k}": v for (m, k), v in W.items()}
say(f"  {'k':>3} {'cells':>8} | {'W0 mean':>8} {'min':>7} {'T449':>6} | {'W1_box':>8} | {'W1_CA':>8} {'max':>8} | smooth frac (min stratum count)")
for k in LEVELS:
    v = np.array([W[(m, k)] for m in P.MODELS])
    say(f"  {k:3d} {1170 * 2 ** k:8.1e} | {v[:, 0].mean():8.4f} {v[:, 0].min():7.4f} {T449_W0.get(k, np.nan):6.3f} | "
        f"{v[:, 1].mean():8.4f} | {np.nanmean(v[:, 2]) if np.isfinite(v[:, 2]).any() else np.nan:8.2e} "
        f"{np.nanmax(v[:, 2]) if np.isfinite(v[:, 2]).any() else np.nan:8.2e} | "
        + " ".join(f"{f:.3f}({c})" for f, c in v[:, 3:5]))
say("\n  ratio = (statistic / 2) / RMS error; 5-model mean (min model)")
for B, e in P.ERR.items():
    say(f"  B = {B:.0e} (error {e:.1e}; cells <= 10 B = {10 * B:.0e})")
    for k in LEVELS:
        v = np.array([W[(m, k)] for m in P.MODELS])
        tag = "within 10B" if 1170 * 2 ** k <= 10 * B else "beyond"
        r = lambda c: f"{np.nanmean(v[:, c]) / 2 / e:9.1f}x ({np.nanmin(v[:, c]) / 2 / e:8.1f}x)"
        say(f"    k {k:2d} {1170 * 2 ** k:8.1e} {tag:>10}: W0 {r(0)} | W1_box {r(1)} | W1_CA {r(2)}")
say("\n  T449 comparability: W0 / 2 / 9.5e-5 (its 10-model RMS at 1e6) at k = 36: mean "
    f"{np.mean([W[(m, 36)][0] for m in P.MODELS]) / 2 / 9.5e-5:.0f}x, min {min(W[(m, 36)][0] for m in P.MODELS) / 2 / 9.5e-5:.0f}x")
# power-law fit of W1_CA over the last three levels, cells needed for 10x
ks = [k for k in LEVELS[-3:]]
y = np.array([np.nanmean([W[(m, k)][2] for m in P.MODELS]) for k in ks])
x = np.log(1170 * 2.0 ** np.array(ks))
if np.isfinite(y).all():
    sl, ic = np.polyfit(x, np.log(y), 1)
    say(f"\n  W1_CA ~ cells^{sl:.3f} (levels {ks}); cells for 10x: " + ", ".join(
        f"B {B:.0e}: {np.exp((np.log(20 * e) - ic) / sl):.1e}" for B, e in P.ERR.items()))
    res["fit_W1CA"] = [sl, ic]
open("analysis.log", "w").write("\n".join(lines_out) + "\n")
json.dump(res, open("analysis.json", "w"), indent=1, default=float)
