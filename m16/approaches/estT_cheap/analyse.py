"""T450 analysis -> analysis.log / analysis.json: per rule max |delta acc|, grade, op ratios; re-scored H-U5 (exact ops /
recipe ops) and H-U6 (recipe MSE ratio vs sampling: err^2 / (p(1-p) / N), N = recipe ops / ops per sample, truth =
exact under the same rule)."""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "understand_m43"))
from zoo44 import ops_per_sample  # noqa: E402

PRE = ["G12", "G16", "G24", "L24", "L32"]
R = {}
for f in glob.glob(os.path.join(HERE, "runs", "H8T*_*.json")):
    if "_sub-" in f:
        continue
    r = json.load(open(f)); R[(r["T"], r["seed"], r["mode"])] = r


def val(r, rule):
    return (r["acc"], r["ops"]) if rule == "REF" else (r["cand"][rule]["acc"], r["cand"][rule]["ops"])


out = {"rules": {}, "HU": {}}
rules = sorted({k for r in R.values() for k in r["cand"]}, key=lambda k: (k not in PRE, k))
lines = []
for rule in rules:
    rs = [r for r in R.values() if rule in r["cand"]]
    md = max(abs(r["cand"][rule]["dacc"]) for r in rs)
    ms = max(r["cand"][rule]["max_dsimplex"] for r in rs)
    orat = [r["cand"][rule]["ops"] / r["ops"] for r in rs]
    grade = "EXACT" if md <= 1e-8 else "ESTIMATE" if md <= 1e-6 else "FAIL"
    cover = sorted({(r["T"], r["seed"], r["mode"][:5]) for r in rs})
    nT = {T: sum(1 for c in cover if c[0] == T) for T in (4, 5)}
    out["rules"][rule] = dict(prereg=rule in PRE, max_dacc=md, max_dsimplex=ms, ops_ratio=[min(orat), max(orat)],
                              grade=grade, runs=nT)
    lines.append(f"{rule:6s} {'pre ' if rule in PRE else 'post'} runs (8,4) {nT[4]:2d} (8,5) {nT[5]:2d} | max |dacc| "
                 f"{md:.1e} | max |dsimplex| {ms:.1e} | op ratio {min(orat):.2f}-{max(orat):.2f} | {grade}")
for T in (4, 5):
    for rule in ["REF"] + [k for k in ("G24", "P5_32", "P5_48") if k in rules]:
        for s in range(5):
            e, c = R.get((T, s, "exact")), R.get((T, s, "recipe108"))
            if e is None or c is None or (rule != "REF" and rule not in e["cand"]):
                continue
            pe, oe = val(e, rule); pc, oc = val(c, rule)
            N = oc / ops_per_sample(8, T); err = pc - pe
            out["HU"][f"{T}_{s}_{rule}"] = d = dict(exact_ops=oe, recipe_ops=oc, HU5=oe / oc, err=err,
                                                    HU6=err ** 2 * N / (pe * (1 - pe)), truth=pe)
            lines.append(f"(8,{T}) seed {s} {rule:6s}: exact {pe:.10f} ops {oe:.3e} | recipe ops {oc:.3e} | H-U5 "
                         f"exact/recipe {d['HU5']:.2f} | recipe err {err:+.2e} | H-U6 MSE ratio {d['HU6']:.3g}")
json.dump(out, open(os.path.join(HERE, "analysis.json"), "w"), indent=1)
open(os.path.join(HERE, "analysis.log"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
