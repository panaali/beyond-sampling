"""T290 summary + pre-registered kill rule; k_prefix=4 regions on 4 models (k in 2,4,6). -> summary.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
D = json.load(open(os.path.join(HERE, "shadow.json"))); keys = list(next(iter(D.values()))["shadows"])
out = {"per_setting": {}}
for k in keys:
    sh = [D[n]["shadows"][k]["units"] < D[n]["units_orig"] or D[n]["shadows"][k]["regions3"] * 4 <= D[n]["regions3_orig"] for n in D]
    ac = [abs(D[n]["shadows"][k]["dacc_pp"]) <= 0.4 for n in D]
    both = np.mean([a and b for a, b in zip(sh, ac)])
    d = np.array([D[n]["shadows"][k]["dacc_pp"] for n in D]); u = np.array([D[n]["shadows"][k]["units"] for n in D])
    rr = np.array([D[n]["shadows"][k]["regions3"] / D[n]["regions3_orig"] for n in D])
    out["per_setting"][k] = dict(frac_shrink=float(np.mean(sh)), frac_acc=float(np.mean(ac)), frac_both=float(both),
                                 median_dacc_pp=float(np.median(d)), max_abs_dacc_pp=float(np.abs(d).max()), units_min=int(u.min()), units_median=float(np.median(u)),
                                 median_region_ratio=float(np.median(rr)))
    print(k, {a: round(b, 3) if isinstance(b, float) else b for a, b in out["per_setting"][k].items()})
out["units_orig"] = sorted({D[n]["units_orig"] for n in D}); out["norm_residual_max"] = max(D[n]["norm_residual"] for n in D)
out["kill"] = not any(v["frac_both"] >= 0.9 for v in out["per_setting"].values()); print("units_orig", out["units_orig"], "KILL" if out["kill"] else "SURVIVES")
if len(sys.argv) > 1:
    import shadow as S
    z = np.load(sys.argv[1], allow_pickle=False); names = list(map(str, z["names"])); k4 = {}
    for nm in ("fused_s0", "fused_s1", "torch_s0", "torch_s1"):
        i = names.index(nm); mn, _ = S.normalise((z["Wi"][i].astype(float), z["Wh"][i].astype(float), z["Wo"][i].astype(float)))
        row = {"orig": S.prefix_run(mn, 4)[0]["regions"]}
        for var in "RG":
            for k in (2, 4, 6):
                ms, u = S.merge(tuple(S.rnd(W, k, var) for W in mn)); row[f"{var}{k}"] = S.prefix_run(ms, 4)[0]["regions"] if u else 1
        k4[nm] = row; print(nm, row, flush=True)
    out["regions4"] = k4
json.dump(out, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
