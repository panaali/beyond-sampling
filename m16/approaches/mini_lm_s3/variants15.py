"""POST-FREEZE exploratory variants (labelled; not part of the pre-registered score) at L = 15, 1e5 FPE, 3 reps.
v_ce20: ce with 20 stages, rho .05; v_top60: top with f_ce .25, f_head .6. -> data/t446/variants15.jsonl"""
import json
import sys
import time
import zlib

import numpy as np

from est import ce_is, toprisk
from s3lib import DATA, Dyck2, load
from taus import points

V = {"v_ce20": lambda m, l, tau, b, rng: ce_is(m, l, 15, tau, b, rng, n_stages=20, rho=0.05),
     "v_top60": lambda m, l, tau, b, rng: toprisk(m, l, 15, tau, b, rng, f_ce=0.25, f_head=0.6)}
out = open(f"{DATA}/variants15.jsonl", "a")
for name in sys.argv[1:]:
    model, _ = load(name); lang = Dyck2(L=15)
    for tau, truth, target in points(name):
        for v, f in V.items():
            for r in range(3):
                t0 = time.time()
                e, c = f(model, lang, tau, 1e5, np.random.default_rng(zlib.crc32(f"{name}|{tau:.6f}|{v}|{r}".encode())))
                out.write(json.dumps(dict(model=name, tau=tau, truth=truth, target=target, budget=1e5, method=v, rep=r, est=e, cost=c, sec=time.time() - t0)) + "\n"); out.flush()
