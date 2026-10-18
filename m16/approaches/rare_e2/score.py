"""T358 scoring: T350's rule; go = (1) >= 2/3 decided wins overall and (2) high-t share >= low-t share - .10."""
import json
import sys
from collections import Counter
import numpy as np

sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/rare_check")
from score_lib import case  # noqa: E402

E, Cn = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
KEY = {"0.5": "E2_0p5", "0.75": "E2_0p75", "1.0": "E2_1", "1.25": "E2_1p25"}
cases, undef = {}, []
for nm, e in E.items():
    n, c = Cn[nm]["n"], Cn[nm]["counts"]
    for t, k in KEY.items():
        p = c[k] / n
        if p == 0:
            undef.append(f"{nm}|{k}"); continue
        cases[f"{nm}|{k}"] = case(e["E2"][t], p, e["C_E2"])
share = lambda sel: (lambda t: dict(t, share=(t["win"] / (t["win"] + t["loss"]) if t["win"] + t["loss"] else None)))(
    Counter(v["outcome"] for k, v in cases.items() if sel(k, v)))
out = {"cases": cases, "undefined": undef, "overall": share(lambda k, v: True),
       "per_t": {k: share(lambda kk, v, k=k: kk.endswith("|" + k)) for k in KEY.values()},
       "low_t": share(lambda k, v: k.endswith("E2_0p5") or k.endswith("E2_0p75")),
       "high_t": share(lambda k, v: k.endswith("E2_1") or k.endswith("E2_1p25")),
       "pC>=1": share(lambda k, v: v["pC"] >= 1)}
rat = {k: [v["ratio"] for kk, v in cases.items() if kk.endswith("|" + k)] for k in KEY.values()}
out["ratio_bands"] = {k: {"within2x": int(sum(.5 <= x <= 2 for x in r)), "within3x": int(sum(1 / 3 <= x <= 3 for x in r)),
                          "n": len(r), "median": float(np.median(r)) if r else None} for k, r in rat.items()}
g1 = out["overall"]["share"] is not None and out["overall"]["share"] >= 2 / 3
g2 = (out["high_t"]["share"] or 0) >= (out["low_t"]["share"] or 0) - .10
out["go"] = {"c1": bool(g1), "c2": bool(g2), "go": bool(g1 and g2)}
json.dump(out, open(sys.argv[3], "w"), indent=1)
for k, v in cases.items():
    print(k, round(v["ratio"], 3), "pC", round(v["pC"], 2), v["outcome"])
print(json.dumps({k: out[k] for k in out if k != "cases"}, indent=1))
