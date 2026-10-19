"""T371 post-hoc analysis (NOT pre-registered): est/truth vs the truth's triple-tie share (hits with g12 < .1), and cost.
usage: python analysis.py (after score.py) -> results/analysis.json"""
import json
import os

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "results", "score.json"))); R = [r for r in S["rows"] if r["hits"] >= 30]
rat = np.array([r["ratio"] for r in R]); tie = np.array([r["tie_share"] for r in R]); ok = (rat >= .5) & (rat <= 2)
out = {}
for lo, hi in ((0, .5), (.5, .8), (.8, 1.01)):
    m = (tie >= lo) & (tie < hi)
    out[f"tie[{lo},{hi})"] = dict(n=int(m.sum()), within2=int(ok[m].sum()), median=float(np.median(rat[m])) if m.any() else None)
out["spearman_tie_vs_log_ratio"] = float(spearmanr(tie, np.log(rat)).correlation)
out["misses"] = [dict(model=r["model"], t0=r["t0"], ratio=r["ratio"], tie=r["tie_share"], truth=r["truth"]) for r, o in zip(R, ok) if not o]
C = {}
for r in S["rows"]:
    C[r["model"]] = r["C_fpe"]
out["C_fpe"] = dict(min=min(C.values()), max=max(C.values()), median=float(np.median(list(C.values()))), total=sum(C.values()))
out["far_share_max"] = max(r["far_share"] or 0 for r in S["rows"])
json.dump(out, open(os.path.join(HERE, "results", "analysis.json"), "w"), indent=1)
for k, v in out.items():
    if k != "misses":
        print(k, v)
for m in out["misses"]:
    print("miss", m)
