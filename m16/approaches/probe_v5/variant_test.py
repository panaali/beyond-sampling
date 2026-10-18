import json, sys
sys.path.insert(0, "/home/user/lb3/m16")
import leaderboard as L
import v5
ms = L.models(); tr = json.load(open("/home/user/lb3/" + L.truths_path(16, 10)))
names = ["orig", "seed1", "seed4", "s0.005_1", "s0.01_1", "s0.02_4"]
for rv, nl in ((1, 7), (1, 3), (2, 3), (1, 1)):
    b = [v5.est_spread(ms[n], 131072 * (1 + rv) // 2, rivals=rv, n_low=nl)[0] - tr[n]["acc"] for n in names]
    print(f"rivals={rv} n_low={nl}:", " ".join(f"{n}:{x:+.4f}" for n, x in zip(names, b)), flush=True)
