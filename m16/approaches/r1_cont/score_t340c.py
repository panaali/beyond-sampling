"""T349: ratios est/truth for T340's variant B (its est.json, pinned) and T340c K = 1, 3, 5, vs T340's GPU truths
(copied read-only to results_t340_truth_gpu_copy.json). usage: python score_t340c.py -> results_t340c_score.json"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
tr = json.load(open(os.path.join(HERE, "results_t340_truth_gpu_copy.json")))
B = json.load(open(os.path.join(os.environ["T340_SRC"], "m16/approaches/rare_events/results/est.json")))
truth = {m: {**{("E1", k): c / tr["E1"][m]["n"] for k, c in tr["E1"][m]["counts"].items()},
             **{("E2", k): c / tr["E2"][m]["n"] for k, c in tr["E2"][m]["counts"].items()}} for m in ("orig", "seed0", "seed3")}
rows = {}; out = {}
for lab in ["B"] + [f"C{K}" for K in (1, 3, 5)]:
    for m in truth:
        if lab == "B":
            e = B[f"{m}:B"]
        else:
            fn = os.path.join(HERE, f"results_t340c_K{lab[1:]}.json")
            if not os.path.exists(fn) or f"{m}:{lab}" not in json.load(open(fn)):
                continue
            e = json.load(open(fn))[f"{m}:{lab}"]
        r = {f"E1({k})": e["est"][k] / truth[m][("E1", k)] for k in ("0.0", "1.0", "1.25", "1.5")}
        r.update({f"E2({k})": e["est2"][k] / truth[m][("E2", k)] for k in ("0.5", "0.75")}); r["C"] = e["C"]
        out[f"{m}:{lab}"] = r
        print(f"{m:6s} {lab:3s} " + "  ".join(f"{k} {v:5.2f}" for k, v in r.items() if k != "C") + f"   C {r['C']:,.0f}")
for lab in ["B", "C1", "C3", "C5"]:
    e1 = [out[f"{m}:{lab}"][f"E1({k})"] for m in truth for k in ("1.0", "1.25", "1.5") if f"{m}:{lab}" in out]
    if len(e1) == 9:
        print(lab, "E1 within [1/2,2]:", sum(.5 <= x <= 2 for x in e1), "/ 9; within [1/3,3]:", sum(1 / 3 <= x <= 3 for x in e1))
json.dump(out, open(os.path.join(HERE, "results_t340c_score.json"), "w"), indent=1)
