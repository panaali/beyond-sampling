"""T370 stage-1 scoring for a variant: E2 (T363's 32 cases, T358 truths), s547 E1 (T367), controls s541 (T367),
s526 (T364), s528-s531 (T366). usage: python score_t370.py VARIANT -> results_t370_<VARIANT>_score.json"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "..", "..", "..", "data", "t363")
var = sys.argv[1]
t358 = json.load(open(os.path.join(D, "t358_rare_mc.json"))); t364 = json.load(open(os.path.join(D, "t364_truth_gpu.json")))
t366 = json.load(open(os.path.join(D, "t366_truth_gpu.json"))); t367 = json.load(open(os.path.join(D, "t367_truth_gpu.json")))
tag = {0.5: "E2_0p5", 0.75: "E2_0p75", 1.0: "E2_1", 1.25: "E2_1p25"}
ld = lambda p: json.load(open(p)) if os.path.exists(p) else {}
e2 = ld(os.path.join(HERE, f"results_t370_{var}_e2.json")); e1 = ld(os.path.join(HERE, f"results_t370_{var}_e1.json"))
out = {"E2": {}, "E1": {}}
for k, r in e2.items():
    nm, t = k.split("|"); p = t358[nm]["counts"][tag[float(t)]] / t358[nm]["n"]; out["E2"][k] = dict(ratio=r["est"] / p, C=r["C"])
def tr(nm, t):
    for src in (t367, t364, t366):
        key = f"{nm}@{float(t):g}"
        if key in src and isinstance(src[key], dict) and "p" in src[key]:
            return src[key]["p"]
for nm, r in e1.items():
    for t, v in r["est"].items():
        p = tr(nm, t); out["E1"][f"{nm}|{t}"] = dict(ratio=v / p if p else None, C=r["C"])
w2 = lambda d: sum(.5 <= v["ratio"] <= 2 for v in d.values() if v["ratio"])
s547 = [v for k, v in out["E1"].items() if k.startswith("dev3_s547")]
ctrl = {k: v for k, v in out["E1"].items() if not k.startswith("dev3_s547")}
out["summary"] = dict(E2_within2x=f"{w2(out['E2'])}/{len(out['E2'])}", s547=[round(v["ratio"], 2) for v in s547],
                      controls_within2x=f"{w2(ctrl)}/{len(ctrl)}",
                      go=bool(len(out["E2"]) == 32 and w2(out["E2"]) >= 24 and len(s547) == 2 and all(.5 <= v["ratio"] <= 2 for v in s547)
                              and len(ctrl) == 12 and w2(ctrl) == 12))
for k, v in sorted(out["E2"].items()):
    print("E2", k, round(v["ratio"], 2))
for k, v in sorted(out["E1"].items()):
    print("E1", k, round(v["ratio"], 2) if v["ratio"] else None)
print(out["summary"]); json.dump(out, open(os.path.join(HERE, f"results_t370_{var}_score.json"), "w"), indent=1)
