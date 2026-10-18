"""T360 scoring: primary = est/truth within [1/2, 2] (>= 80% of 110 cases) and medians in [2/3, 1.5] (all, E1, E2);
secondary = T350's equal-compute rule at executed C. Also a calibration table on T340's models."""
import json
import sys
from collections import Counter
import numpy as np

sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/rare_check")
from score_lib import case  # noqa: E402

E = json.load(open("results/est.json"))
T349 = json.load(open("/home/user/dl/dev3/rare_mc.json")); T358 = json.load(open("/home/user/dl/dev3/t358_rare_mc.json"))
T340 = json.load(open("/home/user/dl/t340/rare_mc.json"))
K1 = {"1.0": "E1_1", "1.25": "E1_1p25", "1.5": "E1_1p5"}
K2 = {"0.5": "E2_0p5", "0.75": "E2_0p75", "1.0": "E2_1", "1.25": "E2_1p25"}


def cases_for(K):
    cs = {}
    for nm, e in E.items():
        if not nm.startswith("dev3"):
            continue
        s = int(nm.split("s")[-1]); src = T349 if s <= 505 else T358
        n, c = src[nm]["n"], src[nm]["counts"]; ek = e[f"K{K}"]
        if s <= 505:
            for t, k in K1.items():
                cs[f"{nm}|{k}"] = case(ek["E1"][t], c[k] / n, ek["C_E1"])
        for t, k in K2.items():
            if k in c and (s > 505 or k in ("E2_0p5", "E2_0p75")):
                cs[f"{nm}|{k}"] = case(ek["E2"][t], c[k] / n, ek["C_E2"])
    return cs


out = {}
for K in (3, 5):
    cs = cases_for(K); r = {k: v["ratio"] for k, v in cs.items()}
    w2 = {k: .5 <= x <= 2 for k, x in r.items()}
    fam = lambda f: [x for k, x in r.items() if f"|{f}" in k]
    per_ev = {ev: {"n": len(v), "within2x": int(sum(.5 <= x <= 2 for x in v)), "median": float(np.median(v))}
              for ev in list(K1.values()) + list(K2.values()) if (v := [x for k, x in r.items() if k.endswith("|" + ev)])}
    med = {"all": float(np.median(list(r.values()))), "E1": float(np.median(fam("E1"))), "E2": float(np.median(fam("E2")))}
    frac = sum(w2.values()) / len(w2)
    t = Counter(v["outcome"] for v in cs.values())
    out[f"K{K}"] = {"n_cases": len(cs), "frac_within2x": frac, "medians": med, "per_event": per_ev,
                    "go": bool(frac >= .8 and all(2 / 3 <= m <= 1.5 for m in med.values())),
                    "equal_compute": dict(t, share=t["win"] / max(t["win"] + t["loss"], 1)),
                    "C_E1_range": [min(E[n][f"K{K}"]["C_E1"] for n in E if n.startswith("dev3_s50") and int(n[-3:]) <= 505),
                                   max(E[n][f"K{K}"]["C_E1"] for n in E if n.startswith("dev3_s50") and int(n[-3:]) <= 505)],
                    "C_E2_range": [min(E[n][f"K{K}"]["C_E2"] for n in E if n.startswith("dev3")),
                                   max(E[n][f"K{K}"]["C_E2"] for n in E if n.startswith("dev3"))],
                    "cases": cs}
cal = {}
for nm in ("orig", "seed0", "seed3"):
    if nm in E:
        n, c = T340[nm]["n"], T340[nm]["counts"]
        cal[nm] = {f"K{K}": {k: E[nm][f"K{K}"][fam][t] / (c[k] / n) for fam, KK in (("E1", K1), ("E2", K2))
                             for t, k in KK.items() if k in c} for K in (3, 5)}
out["calibration_T340_models"] = cal
json.dump(out, open("results/score.json", "w"), indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "cases"} if k.startswith("K") else v for k, v in out.items()},
                 indent=1, default=float))
