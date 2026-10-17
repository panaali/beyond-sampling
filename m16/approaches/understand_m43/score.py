"""Score est_mech against exact truth (arc_m43_repro/results.json) under ARC's protocol."""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "arc_m43_repro"))
import numpy as np
from m43 import Ops, Wi0, Wh0, Wo0, noisy_models, OPS_PER_SAMPLE
from est_mech import est_mech
R = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "arc_m43_repro", "results.json")))
ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="0,1,2,3"); ap.add_argument("--k", type=int, default=256)
ap.add_argument("--strips", default="all"); a = ap.parse_args()
ob = Ops(); ib = {}
print("base: est", est_mech((Wi0, Wh0, Wo0), ob, ib, strips=a.strips), "truth", R["base"]["exact"], "ops", ob.n, "bad", ib["bad_pairs"])
print("  base families:", {k: round(v, 4) for k, v in ib["err_by"].items()})
allerr = []; allpq = []; allops = []
for sd in [int(x) for x in a.seeds.split(",")]:
    ms = noisy_models(256, 0.05, seed=sd)[:a.k]; rows = R["seeds"][str(sd)]["rows"]
    est = []; ops = []; bad = 0
    for m, r in zip(ms, rows):
        o = Ops(); inf = {}; est.append(est_mech(m, o, inf, strips=a.strips)); ops.append(o.n); bad += inf["bad_pairs"] > 0
    tr = np.array([r["truth"] for r in rows[:a.k]]); est = np.array(est); e = est - tr
    allerr += list(e); allpq += list(tr * (1 - tr)); allops += ops
    print(f"seed {sd}: rms {np.sqrt(np.mean(e**2)):.4f} bias {e.mean():+.4f} ops {np.mean(ops):.0f} models_with_bad_pair {bad}")
e = np.array(allerr); pq = np.mean(allpq); ops = np.mean(allops)
mse_s = pq / (ops / OPS_PER_SAMPLE)
print(f"ALL: rms {np.sqrt(np.mean(e**2)):.4f} bias {e.mean():+.4f} ops {ops:.0f} sampling_rmse_at_ops {np.sqrt(mse_s):.4f} MSE_ratio {np.mean(e**2)/mse_s:.4f}")
