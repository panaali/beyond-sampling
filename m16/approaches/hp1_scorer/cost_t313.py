"""T281: apply T313's instrumented cost (r1_v1b2/v1b2.py @ 6a496515: cost(model), and the same count without its B2
first-cell prefix sharing) to R1W V1's unchanged per-model estimates. usage: PANEL=hp1|hp2 python cost_t313.py"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import score  # noqa: E402
V = score.load_sub(os.path.join(score.ROOT, "data/subs/t313_6a496515/m16/approaches/r1_v1b2/v1b2.py"), "cost")
import v1b2, r1w2  # noqa: E402
names, man, models, tr = score.load_panel()
old = {r["k"]: r for r in map(json.loads, open(os.path.join(score.D, "raw_t306_r1w.jsonl")))}
orig = v1b2.distinct_prefix_steps
out = {"t306_r1w_t313b2": [], "t306_r1w_t313nob2": []}
for k, nm in enumerate(names):
    e, _ = r1w2.est(models[k]); assert abs(e - old[k]["est"]) < 1e-12
    v1b2.distinct_prefix_steps = orig; cb2 = v1b2.cost(models[k])
    v1b2.distinct_prefix_steps = lambda x, dx: len(x) * 10     # no sharing: every row runs all T steps
    cno = v1b2.cost(models[k])
    for n_, c in (("t306_r1w_t313b2", cb2), ("t306_r1w_t313nob2", cno)):
        out[n_].append(dict(k=k, name=nm, B=0, est=e, cost=c))
for n_, rows in out.items():
    with open(os.path.join(score.D, f"raw_{n_}.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(n_, score.PANEL, "mean FLOPs %.4g (%.0f FPE)" % (np.mean([r["cost"] for r in rows]), np.mean([r["cost"] for r in rows]) / score.FPE))
    score.aggregate_F(n_, "deductive", (0,))
