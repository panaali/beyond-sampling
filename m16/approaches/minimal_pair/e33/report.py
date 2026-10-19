"""T397 report: MSE ratio vs sampling at executed cost, cluster bootstrap over models, per group."""
import json, sys
import numpy as np
for panel in ("dev3", "lb"):
    try: R = json.load(open(f"e33/scores_{panel}.json"))
    except FileNotFoundError: continue
    grp = (lambda n: "dev3") if panel == "dev3" else (lambda n: "orig" if n == "orig" else "seed" if n.startswith("seed") else n.rsplit("_", 1)[0])
    for b in (16000, 64000):
        for g in sorted({grp(r["model"]) for r in R}):
            S = [r for r in R if r["budget"] == b and grp(r["model"]) == g]
            if not S: continue
            se2 = np.array([(r["est"] - r["truth"]) ** 2 - r["se"] ** 2 for r in S]); v = np.array([r["truth"] * (1 - r["truth"]) / r["cost"] for r in S])
            rng = np.random.default_rng(0); bs = [se2[ii].mean() / v[ii].mean() for ii in (rng.integers(0, len(S), len(S)) for _ in range(2000))]
            print(f"{panel:4s} {g:8s} {b:6d}: MSE ratio {se2.mean()/v.mean():6.3f} [{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}]  "
                  f"exec cost {np.mean([r['cost'] for r in S]):.0f} (claimed {np.mean([r['claimed'] for r in S]):.0f})  bias {100*np.mean([r['est']-r['truth'] for r in S]):+.3f} pp  cpu {np.mean([r['cpu'] for r in S]):.2f}s  n {len(S)}")
