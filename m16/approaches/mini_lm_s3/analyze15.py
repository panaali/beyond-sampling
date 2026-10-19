"""Summarise data/t446/score15.jsonl: per method x budget x architecture group: F_rel (point, bootstrap 95% CI over
models), strict (point <= 1) / lenient (upper CI <= 10), within-2x share, median ratio. -> results/score15.md"""
import json
from collections import defaultdict

import numpy as np

from s3lib import DATA

R = [json.loads(l) for l in open(f"{DATA}/score15.jsonl")]
rng = np.random.default_rng(0)


def frel(rs):
    return [(r["est"] - r["truth"]) ** 2 * max(r["cost"], 1e3) / (r["truth"] * (1 - r["truth"])) for r in rs]


def summary(rs):
    bym = defaultdict(list)
    for r in rs:
        bym[r["model"]].append(r)
    ms = list(bym)
    F = np.mean(frel(rs))
    boots = []
    for _ in range(2000):
        pick = rng.choice(len(ms), len(ms))
        rr = [x for i in pick for x in [bym[ms[i]][j] for j in rng.integers(0, len(bym[ms[i]]), len(bym[ms[i]]))]]
        boots.append(np.mean(frel(rr)))
    lo, hi = np.quantile(boots, [.025, .975])
    ratio = np.array([r["est"] / r["truth"] for r in rs])
    return dict(n_models=len(ms), n_runs=len(rs), F=F, lo=lo, hi=hi, strict=F <= 1, lenient=hi <= 10,
                w2=float(((ratio >= .5) & (ratio <= 2)).mean()), med=float(np.median(ratio)), zero=float((ratio == 0).mean()),
                cost=float(np.median([r["cost"] for r in rs])))


lines = ["| method | budget | group | models | runs | median cost | F_rel [95% CI] | strict F<=1 | lenient CI<=10 | within 2x | median est/truth | share 0 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
keys = sorted({(r["method"], r["budget"]) for r in R}, key=lambda k: (k[0], k[1]))
for meth, b in keys:
    rs = [r for r in R if r["method"] == meth and r["budget"] == b]
    groups = sorted({r["model"].split("_")[1] for r in rs})
    for g in groups + ["ALL"]:
        sub = [r for r in rs if g == "ALL" or r["model"].split("_")[1] == g]
        s = summary(sub)
        lines.append(f"| {meth} | {b:.0e} | {g} | {s['n_models']} | {s['n_runs']} | {s['cost']:.3g} | {s['F']:.3g} [{s['lo']:.3g}, {s['hi']:.3g}] | "
                     f"{'yes' if s['strict'] else 'no'} | {'yes' if s['lenient'] else 'no'} | {100 * s['w2']:.0f}% | {s['med']:.3g} | {100 * s['zero']:.0f}% |")
open("results/score15.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
