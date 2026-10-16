"""QMC convergence rate of the deterministic co-area rule vs T and training seed: spread over R rule seeds
at n faces. usage: python rate_T.py R T:seed ..."""
import sys, json
import numpy as np
import core as C
from est import est_faces
R = int(sys.argv[1]); out = {}
for spec in sys.argv[2:]:
    T, sd_ = spec.split(":"); T = int(T); M = C.zoo(16, T, seed=None if sd_ == "best" else int(sd_))
    for n in (256, 1024, 4096):
        e = []; cs = []
        for r in range(R):
            a, c, _ = est_faces(M, n, n, rule_seed=500 + r); e.append(a); cs.append(c)
        e = np.array(e); p = e.mean(); s = e.std(ddof=1); c = np.mean(cs)
        g = p * (1 - p) / c / s ** 2
        out[f"{spec}_{n}"] = dict(mean=p, sd=s, cost=c, gain=g)
        print(f"T={T} seed={sd_} n={n} acc={p:.5f} sd={1e2*s:.4f}pp cost={c:.0f} gain={g:.1f}", flush=True)
json.dump(out, open("rate_T.json", "w"), indent=1)
