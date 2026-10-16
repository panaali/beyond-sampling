"""Typical error of the deterministic rule: re-run estimator.est with R different fixed rule seeds per model
(each run is still deterministic; the spread shows how lucky one fixed rule is). Reports MSE vs sampling per
group with a bootstrap CI over models x rules. usage: python run_rules.py R budget [budget...]"""
import sys, json, os
import numpy as np
os.chdir("/home/user/clean-neurons")
sys.path.insert(0, "m16"); sys.path.insert(0, "m16/approaches/sci_tao")
import leaderboard as LB
import estimator as E
R = int(sys.argv[1]); truth = json.load(open(LB.TRUTHS)); ms = LB.models(); out = {}
for B in map(int, sys.argv[2:]):
    for name, m in ms.items():
        p = truth[name]["acc"]; rows = []
        for r in range(R):
            E.RULE_SEED = 3000 + 17 * r
            a, c = E.est(m, B); rows.append((a - p, p * (1 - p) / c))
        out[f"{name}|{B}"] = rows
    rng = np.random.default_rng(0)
    for grp in ("orig", "seed", "s0.0025", "s0.005", "all"):
        ks = [k for k in out if k.endswith(f"|{B}") and (grp == "all" or k.startswith(grp))]
        e2 = np.array([[x[0] ** 2 for x in out[k]] for k in ks]); sv = np.array([[x[1] for x in out[k]] for k in ks])
        bias = np.array([[x[0] for x in out[k]] for k in ks])
        ratio = e2.mean() / sv.mean()
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(ks), len(ks)); j = rng.integers(0, R, R)
            bs.append(e2[i][:, j].mean() / sv[i][:, j].mean())
        print(f"B={B} {grp:8s} models={len(ks)} RMSE={1e2*np.sqrt(e2.mean()):.3f}pp sampling={1e2*np.sqrt(sv.mean()):.3f}pp "
              f"MSE ratio={ratio:.2f} [{np.percentile(bs,2.5):.2f},{np.percentile(bs,97.5):.2f}] mean err={1e2*bias.mean():+.3f}pp", flush=True)
json.dump(out, open(f"m16/approaches/sci_tao/rules_{R}.json", "w"))
