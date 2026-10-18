"""T354: score the fixed-walk reruns (B, C3, C5, S3, S5) vs T340's GPU truths; F vs sampling at equal compute.
-> results_fixed_score.json"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
tr = json.load(open(os.path.join(HERE, "results_t340_truth_gpu_copy.json")))
R = {}
for fn in glob.glob(os.path.join(HERE, "results_fixed_*.json")):
    if fn.endswith("_score.json"):
        continue
    R.update(json.load(open(fn)))
ev = [("E1", "1.0"), ("E1", "1.25"), ("E1", "1.5"), ("E2", "0.5"), ("E2", "0.75")]
out = {}
for tag in ("B", "C3", "C5", "S3", "S5"):
    for m in ("orig", "seed0", "seed3"):
        key = f"{m}:{tag}"
        if key not in R:
            continue
        r = R[key]; C = r["C"]; row = {"C": C}
        for e, k in ev:
            p = tr[e][m]["counts"][k] / tr[e][m]["n"]; est = (r["est"] if e == "E1" else r["est2"])[k]
            row[f"{e}({k})"] = dict(ratio=est / p, F=(est - p) ** 2 * max(C, 1000) / (p * (1 - p)), pC=p * C)
        out[key] = row
        print(f"{m:6s} {tag:3s} C {C:10,.0f}  " + "  ".join(f"{k} {v['ratio']:5.2f} (F {v['F']:6.2f})" for k, v in row.items() if k != "C"))
    e1 = [out[f"{m}:{tag}"][f"E1({k})"]["ratio"] for m in ("orig", "seed0", "seed3") for k in ("1.0", "1.25", "1.5") if f"{m}:{tag}" in out]
    e2 = [out[f"{m}:{tag}"][f"E2({k})"]["ratio"] for m in ("orig", "seed0", "seed3") for k in ("0.5", "0.75") if f"{m}:{tag}" in out]
    if len(e1) == 9:
        print(f"  {tag}: E1 within 2x {sum(.5 <= x <= 2 for x in e1)}/9, within 3x {sum(1/3 <= x <= 3 for x in e1)}/9; E2 within 2x {sum(.5 <= x <= 2 for x in e2)}/6")
json.dump(out, open(os.path.join(HERE, "results_fixed_score.json"), "w"), indent=1)
