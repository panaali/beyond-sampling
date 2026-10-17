"""T108 truth check at (8,4): per half-chamber, estT exact-mode error mass vs MC (2e6 inputs); locates geometry bugs.
usage: python check108.py [H] [seed]"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "arc_m43_repro"))
import numpy as np
import estT
from m43 import Ops
from zoo44 import zoo

H = int(sys.argv[1]) if len(sys.argv) > 1 else 8; sd = int(sys.argv[2]) if len(sys.argv) > 2 else 0
m = zoo(H, 4, sd); Wi, Wh, Wo = m
Hs = estT.halves(4)
x = np.random.default_rng(0).standard_normal((2_000_000, 4))
h = np.zeros((len(x), H))
for t in range(4):
    h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
wrong = (h @ Wo.T).argmax(1) != np.argsort(x, 1)[:, -2]
rows = []; estT.MODE[0] = "exact"
for hh in Hs:
    ins = np.all(x @ hh["cone"].N.T >= 0, 1); mc = np.mean(ins & wrong)
    g1 = (hh["sigma"] * Wi > 0).astype(float)
    e = estT._finish(hh["cone"], g1, np.outer(Wi, np.eye(4)[0]), 1, m, hh["truth"], hh["n_true"], 0.1, Ops(), True)
    rows.append((e - mc, mc, e, hh["perm"], hh["sigma"], tuple(hh["n_true"])))
rows.sort(key=lambda r: -abs(r[0]))
for r in rows[:6]:
    print(r, flush=True)
print("total exact err", sum(r[2] for r in rows), "mc err", sum(r[1] for r in rows), "mc overall", wrong.mean())
