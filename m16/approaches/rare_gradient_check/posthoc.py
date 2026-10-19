"""Post hoc (NOT pre-registered): rescore G1 against a small-step truth = inverse-variance mean of the eps .003 and
.001 levels (curvature bias O(eps^2) negligible there). usage: python posthoc.py -> results/posthoc.txt"""
import json
import numpy as np
S = json.load(open("results/score.json")); L = []; k = n = 0
for nm in ("orig", "dev3_s528"):
    T = json.load(open(f"results/truth_{nm}.json"))
    for r, s in zip(T["rows"], S[nm]):
        if r["name"].startswith("interv"):
            continue
        lv = r["levels"][2:]; w = np.array([1 / l["se"] ** 2 for l in lv]); m = float((w * [l["dlog"] for l in lv]).sum() / w.sum()); se = float(w.sum() ** -.5)
        res = abs(m) > 3 * se; ok = res and np.sign(s["est"]) == np.sign(m) and .5 <= s["est"] / m <= 2
        k += ok; n += res
        L.append(f"{nm:10s} {r['name']:10s} small-step truth {m:7.1f} ± {se:4.1f}  est {s['est']:7.1f}  {'pass' if ok else ('FAIL' if res else 'unres.')}")
L.insert(0, f"post hoc G1 vs small-step truths: {k}/{n} resolved pass")
open("results/posthoc.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
