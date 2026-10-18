"""v5a bias on all 30 LB3 models at large N (bias-dominated)."""
import json
import sys
import numpy as np
sys.path.insert(0, "/home/user/lb3/m16")
import leaderboard as L
import v5

N = int(sys.argv[1])
ms = L.models()
tr = json.load(open("/home/user/lb3/" + L.truths_path(16, 10)))
out = {}
for name, m in ms.items():
    e, _ = v5.est(m, N)
    out[name] = e - tr[name]["acc"]
    print(name, "%.5f" % out[name], flush=True)
json.dump(out, open(f"results/bias_scan_{N}.json", "w"), indent=1)
