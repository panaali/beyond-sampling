"""Deterministic evaluation on all leaderboard models of M_16,T (one run per model: the method has no randomness).
Reports, per group and overall: bias, RMSE (pp), cost as implemented and at Hilton's O(m D^2)-per-cut accounting
(both in FPE), and the MSE ratio to plain sampling at each cost (v3 rule: mean p(1-p)/cost; below 1 beats it).
usage: python eval_all.py T D [D ...]      -> eval_T{T}.json (appends)"""
import json, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../.."))
from leaderboard import models, group_of, truths_path, fpe_flops
import hprop

T = int(sys.argv[1]); Ds = [int(a) for a in sys.argv[2:]]; H = 16
tp = os.path.join(HERE, f"results_truths_H{H}_T{T}.json")
truth = json.load(open(tp if os.path.exists(tp) else os.path.join(HERE, "../../..", truths_path(H, T))))
out_p = os.path.join(HERE, f"eval_T{T}.json")
out = json.load(open(out_p)) if os.path.exists(out_p) else {}
F = fpe_flops(H, T)
for D in Ds:
    rows = {}
    for name, m in models(H, T).items():
        t0 = time.process_time()
        e, info = hprop.estimate(m, D, cells="first", return_info=True)
        rows[name] = dict(est=e, truth=truth[name]["acc"], se=truth[name]["se"], cost=info["flops"] / F,
                          cost_hilton=info["hilton_flops"] / F, cost_lp=info["lp_flops"] / F,
                          ncut=info["ncut"], nleaf=info["nleaf"], cpu=time.process_time() - t0)
    out[str(D)] = rows
    json.dump(out, open(out_p, "w"), indent=1)
    print(f"\nT={T} D={D}")
    for g in ["seed", "s0.0025", "s0.005", "s0.01", "s0.02", "all"]:
        sel = [r for k, r in rows.items() if g == "all" or group_of(k) == g]
        if not sel:
            continue
        err = np.array([r["est"] - r["truth"] for r in sel]); se2 = np.array([r["se"] ** 2 for r in sel])
        mse = max((err ** 2 - se2).mean(), 1e-30)
        pq = np.array([r["truth"] * (1 - r["truth"]) for r in sel])
        c = np.array([r["cost"] for r in sel]); ch = np.array([r["cost_hilton"] for r in sel])
        print(f"  {g:8s} n={len(sel):2d} bias {100*err.mean():+8.3f} pp  RMSE {100*np.sqrt(mse):7.3f} pp  "
              f"cost {c.mean():9.3g} FPE (Hilton-acct {ch.mean():9.3g}; cones alone {np.mean([r['cost_lp'] for r in sel]):8.3g})  "
              f"MSE/sampling: {mse/(pq/c).mean():9.3g} (as impl.)  {mse/(pq/ch).mean():9.3g} (Hilton acct)  "
              f"leaves {np.mean([r['nleaf'] for r in sel]):.0f}", flush=True)
