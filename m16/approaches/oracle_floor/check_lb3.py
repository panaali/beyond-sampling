"""Check that the reconstructed LB3 models match results/leaderboard_truths.json (2e6 fresh samples each)."""
import json, os
import numpy as np
from panel import ROOT, lb3, forward
tr = json.load(open(os.path.join(ROOT, "results/leaderboard_truths.json"))); bad = 0
for k, (n, m) in enumerate(lb3()):
    rng = np.random.default_rng(296000 + k); ok = 0
    for _ in range(4):
        x = rng.standard_normal((500_000, 10)); ok += (forward(m, x).argmax(1) == np.argsort(x, 1)[:, -2]).sum()
    a = ok / 2e6; z = (a - tr[n]["acc"]) / np.sqrt(a * (1 - a) / 2e6 + tr[n]["se"] ** 2); bad += abs(z) > 4
    print(n, "%.5f vs LB3 %.5f  z %+.2f" % (a, tr[n]["acc"], z), flush=True)
print("mismatches:", bad)
