"""T323: exact DEV2 F for T317c from its per-model DEV2 estimates and the open DEV2 truths.
Inputs (git show into files): claude/t218-tie-tail-labels m16/approaches/r1_cont/results_d.json (est_c = T317c),
claude/t192-gpu-cost-law m16/approaches/gpu_cost_law/hp2/dev2/truths.json.
Usage: python dev2_f.py results_d.json truths.json"""
import json
import sys
import numpy as np

d = json.load(open(sys.argv[1]))["rows"]
t = json.load(open(sys.argv[2]))["truths"]
ks = sorted(d)
e = np.array([d[k]["est_c"] for k in ks])
p = np.array([t[k]["acc"] for k in ks])
err = e - p
mse, pq = (err ** 2).mean(), (p * (1 - p)).mean()
print(f"n={len(ks)} bias={err.mean()*100:.3f}pp rms={np.sqrt(mse)*100:.3f}pp mean_pq={pq:.5f} n*={pq/mse:.0f}")
rng = np.random.default_rng(323)
for C in (5435.188, 5488.0):          # 1c742430 frozen count; c70e22cd as executed
    bs = []
    for _ in range(10000):            # one cluster per model
        i = rng.integers(0, len(ks), len(ks))
        bs.append((err[i] ** 2).mean() * C / (p[i] * (1 - p[i])).mean())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"C={C:.0f} F={mse*C/pq:.2f} [{lo:.2f}, {hi:.2f}]")
