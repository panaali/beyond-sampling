"""Fixed rule on the 18 leaderboard models vs leaderboard truths; plus sampling's expected RMSE at equal cost.
run from repo root: python m16/approaches/sci_feynman/eval_all.py budget"""
import sys, json
import numpy as np
sys.path.insert(0, "m16"); sys.path.insert(0, "m16/approaches/sci_feynman")
import leaderboard as L
import estimator as E
b = int(sys.argv[1]); truth = json.load(open(L.TRUTHS)); out = {}
for name, m in L.models().items():
    e, c = E.est(m, b, None); t = truth[name]["acc"]
    out[name] = dict(est=e, cost=c, truth=t, err_pp=100 * (e - t), samp_sd_pp=100 * np.sqrt(t * (1 - t) / c))
    print(f"{name:10s} est {e:.5f} truth {t:.5f} err {100*(e-t):+.3f}pp  sampling sd {out[name]['samp_sd_pp']:.3f}pp cost {c:.0f}", flush=True)
for g in ("orig", "seed", "s0.0025", "s0.005"):
    ks = [k for k in out if k.startswith(g)]
    mse = np.mean([out[k]["err_pp"] ** 2 for k in ks]); ms = np.mean([out[k]["samp_sd_pp"] ** 2 for k in ks])
    print(f"group {g:8s} RMSE {np.sqrt(mse):.3f}pp  sampling RMSE {np.sqrt(ms):.3f}pp  MSE ratio {mse/ms:.2f}  (n={len(ks)})")
json.dump(out, open(f"m16/approaches/sci_feynman/eval_{b}.json", "w"), indent=1)
