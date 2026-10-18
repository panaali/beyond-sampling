"""Cost sanity check (leaderboard rule 6): CPU seconds of each sweep row / (charged FPE x plain sampling's CPU
seconds per FPE at batch 1000). A ratio far above 1 flags under-counted cost (or Python overhead)."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cf  # noqa: E402
import leaderboard as L  # noqa: E402

m = cf.lb3_models()["orig"]; rng = np.random.default_rng(0); t = time.process_time()
for _ in range(200):
    L.est_sampling(m, 1000, rng)
spf = (time.process_time() - t) / 200_000
rows = [json.loads(l) for l in open(os.path.join(cf.HERE, "results", "sweep_raw.jsonl"))]
print(f"plain sampling: {spf * 1e6:.2f} us per FPE (batch 1000)")
for fam in ("rx", "obs", "probe", "rel"):
    r = np.array([d["cpu_s"] / (d["cost_fpe"] * spf) for d in rows if d["fam"] == fam])
    print(f"{fam:6s} time ratio median {np.median(r):6.2f}  [p10 {np.percentile(r, 10):.2f}, p90 {np.percentile(r, 90):.2f}]")
