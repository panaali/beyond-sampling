"""T398 add-on: pruning on bench_182_k4 nodes as a function of the normalisation DELTA (g23 >= DELTA ||x||_inf).
Rigorous at threshold t0 if 20 Phi(-t0/DELTA) is negligible: t0 = 2.5 -> DELTA = 0.33 loses 6e-13."""
import sys

import numpy as np

import relax2

d = np.load("nodes/bench_182_k4.npz", allow_pickle=True)
tri = tuple(int(v) for v in d["tri"]); k = int(d["k"])
rng = np.random.default_rng(0); idx = rng.choice(len(d["m"]), 60, replace=False)
for delta in [float(a) for a in sys.argv[1].split(",")]:
    relax2.DELTA = delta; st = []
    for i in idx:
        G = d["G"][i][:d["m"][i]]
        st.append((relax2.prune_stage(G, d["A"][i], k, tri, depth=0), bool(d["is_event"][i])))
    st = np.array(st, dtype=object)
    pr = np.array([s >= 0 for s, _ in st]); ev = np.array([e for _, e in st])
    print(f"DELTA {delta}: pruned {pr.sum()}/{len(pr)} (by norm {sum(s == 0 for s, _ in st)}); "
          f"line-walk event nodes pruned {(pr & ev).sum()}/{ev.sum()}", flush=True)
