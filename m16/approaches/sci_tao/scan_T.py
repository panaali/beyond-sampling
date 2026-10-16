"""Structure->randomness transition in sequence length T: MSE gain of the deterministic co-area rule over
sampling at equal compute, from the spread over R fixed rule seeds. usage: python scan_T.py n R"""
import sys, json
import numpy as np
import core as C
from est import est_faces
sys.path.insert(0, "/home/user/clean-neurons/m16")
n = int(sys.argv[1]); R = int(sys.argv[2])
out = {}
models = [(f"M16_{T}", C.zoo(16, T)) for T in (3, 4, 5, 6, 8)]
from common import Wi, Wh, Wo
models.append(("orig", (Wi, Wh, Wo)))
for name, M in models:
    est = []; costs = []; cv = {}; cells = []
    for rs in range(R):
        a, c, o = est_faces(M, n, n, rule_seed=1000 + rs)
        est.append(a); costs.append(c)
        for k, v in o.items():
            cv.setdefault(k, []).append(v.std() / max(v.mean(), 1e-12))
    est = np.array(est); p = est.mean(); cost = np.mean(costs)
    sd = est.std(ddof=1)
    gain = p * (1 - p) / cost / sd ** 2
    out[name] = dict(mean=p, sd=sd, cost=cost, gain=gain, cv={k: float(np.mean(v)) for k, v in cv.items()})
    print(f"{name:7s} est={p:.5f} sd={1e2*sd:.4f}pp cost/run={cost:.0f} samp_sd={1e2*np.sqrt(p*(1-p)/cost):.4f}pp "
          f"MSE gain={gain:.2f}  CV(F): {out[name]['cv']}", flush=True)
json.dump(out, open(f"scan_T_{n}_{R}.json", "w"), indent=1)
