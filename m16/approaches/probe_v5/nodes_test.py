"""Bias of gap-node probes vs background treatment (many nodes, so bias dominates noise).
A: top-3 relative gaps from the exact (model-free) joint law, ranks 4..10 hung below v3 at canonical EOS spacings (shuffled);
B: whole order-statistic vector from the model-free law (= stratified sampling; unbiased control).
Each node: the network's answer on the constructed input, correct iff it is the rank-2 position."""
import sys
import numpy as np
sys.path.insert(0, "/home/user/lb3/m16")
import leaderboard as L
from probes import TRIPLES, forward, EOS, _ref

ms = L.models()
tr = __import__("json").load(open("/home/user/lb3/" + L.truths_path(16, 10)))
NODES = 40
rng = np.random.default_rng(11)
rows = _ref[rng.choice(len(_ref), NODES * 720, replace=False)]          # model-free order-stat vectors
nb = rows[:, 1] > 0                                                      # non-blind part only
for name in sys.argv[1:]:
    m = ms[name]
    res = {}
    for mode in ("A", "B"):
        X = np.empty((NODES * 720, 10))
        for i in range(NODES * 720):
            t = TRIPLES[i % 720]
            v = rows[i].copy()
            if mode == "A":
                v[3:] = v[2] - (EOS[2] - EOS[3:])      # canonical spacings below the real v3
            rest = [p for p in range(10) if p not in t]
            r = np.arange(3, 10); np.random.default_rng(i).shuffle(r)
            X[i, t] = v[:3]; X[i, rest] = v[r]
        ok = forward(m, X) == TRIPLES[np.arange(len(X)) % 720, 1]
        res[mode] = 1 - ok[nb].mean()        # non-blind error rate
    print(name, "nonblind err A %.4f  B %.4f  (A-B %.4f)" % (res["A"], res["B"], res["A"] - res["B"]), flush=True)
