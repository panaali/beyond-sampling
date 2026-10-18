"""T274b step 0: re-score leaderboard-v3 rows at EVERY budget with cluster-bootstrap CIs (same rule as
m16/leaderboard.py table(): per-run sampling MSE p(1-p)/charged cost, bootstrap over models, 2000 draws, seed 0).
Inputs: lb3_raw_snapshot.jsonl and lb3_truths.json, copied from origin/ccr-6a61f25c-wn80ii (results/).
usage: python rescore.py NAME [cost_factor]"""
import json
import sys

import numpy as np

name = sys.argv[1]; cf = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
truth = json.load(open("lb3_truths.json"))
rows = {}
for l in open("lb3_raw_snapshot.jsonl"):
    d = json.loads(l)
    if d["v"] == 3 and d["H"] == 16 and d["T"] == 10 and d["name"] == name:
        rows[(d["model"], d["budget"], d["rep"])] = d
rs = list(rows.values())
grp = lambda m: "orig" if m == "orig" else "seed" if m.startswith("seed") else m.rsplit("_", 1)[0]
G = ["orig", "seed", "s0.0025", "s0.005", "s0.01", "s0.02"]
print(f"{name}: ratio = estimator MSE / sampling MSE at equal charged cost [95% cluster-bootstrap CI]; bias pp")
print("budget  " + "  ".join(f"{g:>22s}" for g in G))
for b in sorted({d["budget"] for d in rs}):
    cells = []
    for g in G:
        sel = [d for d in rs if d["budget"] == b and grp(d["model"]) == g]
        if not sel:
            cells.append(" " * 22); continue
        se2 = np.array([(d["est"] - truth[d["model"]]["acc"]) ** 2 - truth[d["model"]]["se"] ** 2 for d in sel])
        cst = np.array([d["cost"] * cf for d in sel])
        v = np.array([truth[d["model"]]["acc"] * (1 - truth[d["model"]]["acc"]) for d in sel]) / cst
        mods = sorted({d["model"] for d in sel}); by = {m: np.flatnonzero([d["model"] == m for d in sel]) for m in mods}
        br = np.random.default_rng(0); bs = []
        for _ in range(2000):
            ii = np.concatenate([br.choice(by[m], len(by[m])) for m in br.choice(mods, len(mods))])
            bs.append(se2[ii].mean() / v[ii].mean())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        bias = 100 * np.mean([d["est"] - truth[d["model"]]["acc"] for d in sel])
        cells.append(f"{se2.mean() / v.mean():7.2f} [{lo:6.2f},{hi:6.2f}] {bias:+5.2f}"[:22].rjust(22))
    print(f"{b:>7d} " + "  ".join(cells))
