import json, sys
import numpy as np
sys.path.insert(0, "/home/user/lb3/m16")
import leaderboard as L
import v5
B = int(sys.argv[1])
ms = L.models(); tr = json.load(open("/home/user/lb3/" + L.truths_path(16, 10)))
out = {}
for name, m in ms.items():
    e, c = v5.est_spread(m, B)
    out[name] = e - tr[name]["acc"]
    print(name, "%.5f" % out[name], flush=True)
json.dump(out, open(f"results/bias_scan_b_{B}.json", "w"), indent=1)
