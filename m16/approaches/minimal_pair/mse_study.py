"""Ablation: MSE over R fixed rule seeds of 4 variants x budgets, zoo(16,T) seeds 0-4. Raw estimates -> mse_T{T}.json.
Variants: plain (sci_tao face law, Gaussian integrand, kind-level Neyman), radial, strata, radial+strata (= frozen)."""
import json, sys, time
import numpy as np
import core as C, estimator as E
T = int(sys.argv[1]); R = int(sys.argv[2]); budgets = [int(b) for b in sys.argv[3].split(",")]
V = {"plain": dict(radial=False, stratify=False), "radial": dict(radial=True, stratify=False),
     "strata": dict(radial=False, stratify=True), "frozen": dict()}
if len(sys.argv) > 4:
    V = {k: V[k] for k in sys.argv[4].split(",")}
out = {}
for s in range(5):
    m = C.zoo(16, T, s)
    for v, kw in V.items():
        for b in budgets:
            t0 = time.process_time()
            r = [E.est(m, b, rule_seed=7000 + 13 * k, **kw) for k in range(R)]
            out[f"{s}|{v}|{b}"] = dict(est=[e for e, _ in r], cost=[c for _, c in r], cpu=time.process_time() - t0)
            print(T, s, v, b, f"mean {np.mean([e for e,_ in r]):.6f} sd {np.std([e for e,_ in r]):.2e} cost {np.mean([c for _,c in r]):.0f} cpu {time.process_time()-t0:.1f}", flush=True)
    json.dump(out, open(f"mse_T{T}.json", "w"))
