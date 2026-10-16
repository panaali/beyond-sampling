"""Evaluate estimator.est on the 18 leaderboard models vs cached 4M-sample truths.
usage: python run_eval.py budget [budget ...]  -> eval_<budget>.json"""
import sys, json, time, os
import numpy as np
os.chdir("/home/user/clean-neurons")
sys.path.insert(0, "m16"); sys.path.insert(0, "m16/approaches/sci_tao")
import leaderboard as LB
import estimator as E
truth = json.load(open(LB.TRUTHS)); ms = LB.models()
for B in map(int, sys.argv[1:]):
    rows = {}
    for name, m in ms.items():
        t0 = time.time(); a, c = E.est(m, B)
        p = truth[name]["acc"]
        rows[name] = dict(est=a, cost=c, truth=p, err=a - p, samp_sd=float(np.sqrt(p * (1 - p) / c)))
        print(f"B={B} {name:10s} est={a:.5f} truth={p:.5f} err={1e2*(a-p):+.3f}pp cost={c:.0f} samp_sd={1e2*rows[name]['samp_sd']:.3f}pp ({time.time()-t0:.0f}s)", flush=True)
    json.dump(rows, open(f"m16/approaches/sci_tao/eval_{B}.json", "w"), indent=1)
    for grp in ("orig", "seed", "s0.0025", "s0.005"):
        ks = [k for k in rows if k.startswith(grp)]
        e = np.array([rows[k]["err"] for k in ks]); s = np.array([rows[k]["samp_sd"] for k in ks])
        print(f"B={B} {grp:8s} n={len(ks)} RMSE={1e2*np.sqrt((e**2).mean()):.3f}pp sampling RMSE={1e2*np.sqrt((s**2).mean()):.3f}pp MSE ratio(est/samp)={(e**2).mean()/(s**2).mean():.2f} mean bias={1e2*e.mean():+.3f}pp")
