"""T350: score estimates against a counts file (T192 rare_mc format). Usage: score.py est.json counts.json out.json"""
import json
import sys
from collections import Counter
from score_lib import case

E, Cn = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
K1 = {"1.0": "E1_1", "1.25": "E1_1p25", "1.5": "E1_1p5"}; K2 = {"0.5": "E2_0p5", "0.75": "E2_0p75"}
out = {"cases": {}, "tally": {}}
for nm, e in E.items():
    n = Cn[nm]["n"]; c = Cn[nm]["counts"]
    for v in ("A", "B"):
        for fam, K in (("E1", K1), ("E2", K2)):
            for t, k in K.items():
                out["cases"][f"{nm}|{v}|{k}"] = case(e[v][fam][t], c[k] / n, e[v][f"C_{fam}"])
    out["cases"][f"{nm}|cal|E1_0"] = case(e["B"]["E1"]["0.0"], c["E1_0"] / n, e["B"]["C_E1"])
for label, sel in (("A", lambda k: "|A|" in k), ("B", lambda k: "|B|" in k),
                   ("family", lambda k: ("|B|E1" in k) or ("|A|E2" in k)),
                   ("family_E1", lambda k: "|B|E1" in k), ("family_E2", lambda k: "|A|E2" in k),
                   ("family_pC>=1", lambda k: (("|B|E1" in k) or ("|A|E2" in k)) and out["cases"][k]["pC"] >= 1)):
    t = Counter(v["outcome"] for k, v in out["cases"].items() if sel(k))
    nt = t["win"] + t["loss"]
    out["tally"][label] = dict(t, frac_win_nontied=(t["win"] / nt if nt else None))
fam = out["tally"]["family"]; out["go"] = bool(fam["frac_win_nontied"] is not None and fam["frac_win_nontied"] >= 2 / 3)
json.dump(out, open(sys.argv[3], "w"), indent=1)
for k, v in out["cases"].items():
    print(k, round(v["ratio"], 3), "pC", round(v["pC"], 2), "r", round(v["r"], 3), "s", round(v["s"], 3), v["outcome"])
print(json.dumps(out["tally"], indent=1), "GO" if out["go"] else "no go")
