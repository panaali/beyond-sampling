"""Summarise results/mix.json: per-unit K* per rule, best-rule K*, pass rates at K <= 50, verdict."""
import json, os
import numpy as np
from mix import LADDER, passes
HERE = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(os.path.join(HERE, "results/mix.json")))
RULES = ("naive", "future", "future_gates", "future_gates_X", "pre_future_gates_X")
BIG = 10 ** 6
def kstar(r, rule):
    return next((K for K in LADDER if passes(r[rule][str(K)], r["P"])), BIG)
def kstable(r, rule):          # post hoc: smallest K from which every larger ladder rung passes
    ok = [passes(r[rule][str(K)], r["P"]) for K in LADDER]
    return next((K for i, K in enumerate(LADDER) if all(ok[i:])), BIG)
out = {}
for kind in ("blind", "class", "all"):
    R = [r for r in res.values() if kind == "all" or r["kind"] == kind]
    d = {"n": len(R)}
    for rule in RULES + ("best",):
        ks = [min(kstar(r, q) for q in RULES) if rule == "best" else kstar(r, rule) for r in R]
        ks = np.array(ks)
        d[rule] = dict(median_Kstar=(">128" if np.median(ks) >= BIG else float(np.median(ks))), pass_le50=float((ks <= 50).mean()),
                       pass_le128=float((ks <= 128).mean()), pass_K1=float((ks <= 1).mean()))
    ksb = np.array([min(kstable(r, q) for q in RULES) for r in R])
    d["stable_best_posthoc"] = dict(median=(">128" if np.median(ksb) >= BIG else float(np.median(ksb))), pass_le50=float((ksb <= 50).mean()))
    best_err = lambda K: [min(abs(r[q][str(K)] - r["P"]) / max(min(r["P"], 1 - r["P"]), 1e-9) for q in RULES) for r in R]
    d["median_rel_err_best_rule"] = {K: float(np.median(best_err(K))) for K in LADDER}
    out[kind] = d
b = out["all"]["best"]; med = b["median_Kstar"]
out["verdict"] = "STOP (dead in the cheap regime, scoped)" if med == ">128" or med > 50 else ("GO signal" if b["pass_le50"] >= .8 else "inconclusive")
json.dump(out, open(os.path.join(HERE, "results/mix_summary.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
