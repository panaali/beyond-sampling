"""T455 cost estimate for all late triples (second-smallest position >= 7; 132 ordered triples). Picks 12 triples by
the pre-registered rule: 2 per (second-smallest position b in {7, 8}) x (rank of p3 among the three positions:
smallest / middle / largest), the first two of each stratum in a fixed shuffle (seed 455). Runs T418's dyadic-leaf
estimator (alpha-CROWN, n = 500) on each.
usage: python latecost.py list | python latecost.py run OUTDIR"""
import itertools
import os
import subprocess
import sys

import numpy as np

late = [t for t in itertools.permutations(range(10), 3) if sorted(t)[1] >= 7]
assert len(late) == 132
strata = {}
for t in late:
    s = sorted(t); key = (s[1], s.index(t[2]))
    strata.setdefault(key, []).append(t)
rng = np.random.default_rng(455); pick = []
for key in sorted(strata):
    L = strata[key]; perm = rng.permutation(len(L)); pick += [L[i] for i in perm[:2]]
if sys.argv[1] == "list":
    print(len(late), {k: len(v) for k, v in sorted(strata.items())}); print(pick)
else:
    D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "late_triple_verify", "dyadic.py"); od = sys.argv[2]
    for t in pick:
        name = os.path.join(od, "d2_%d%d%d.json" % t)
        if not os.path.exists(name):
            subprocess.run([sys.executable, D, "%d,%d,%d" % t, name, "500", "alpha-CROWN"], check=False)
