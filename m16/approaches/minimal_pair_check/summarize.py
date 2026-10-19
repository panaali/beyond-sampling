"""Group table from score.json + truths.json: per sigma group, MSE ratio = mean(se2) / mean(p(1-p)/cost) over the
10 noisy copies (5 fresh models x 2), with a bootstrap 95% CI over models (one deterministic run per model).
usage: python summarize.py"""
import json

import numpy as np

S = json.load(open("score.json"))["runs"]; TR = json.load(open("truths.json"))
groups = {}
for k, r in S.items():
    name, b = k.split("|")
    g = "clean" if "/" not in name else name.split("/")[1].rsplit("_", 1)[0]
    p, se = TR[name]["acc"], TR[name]["se"]
    groups.setdefault((g, int(b)), []).append(((r["est"] - p) ** 2 - se ** 2, p * (1 - p) / r["cost"], p))
rng = np.random.default_rng(0)
for (g, b), v in sorted(groups.items()):
    v = np.array(v); bi = rng.integers(0, len(v), (4000, len(v)))
    boot = v[bi, 0].mean(1) / v[bi, 1].mean(1)
    print(f"{g:8s} {b:6d}  n={len(v):2d}  acc {v[:, 2].min():.3f}-{v[:, 2].max():.3f}  ratio {v[:, 0].mean() / v[:, 1].mean():6.3f} "
          f"[{np.percentile(boot, 2.5):6.3f}, {np.percentile(boot, 97.5):6.3f}]  per-model max {(v[:, 0] / v[:, 1]).max():6.2f}"
          f"  #>=1: {(v[:, 0] / v[:, 1] >= 1).sum()}")
