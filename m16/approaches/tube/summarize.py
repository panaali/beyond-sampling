"""RMSE of the deterministic tube estimate vs truth (4M samples, ../debiased_series/truth.json, identical to
arc_eval.true_acc(m, n=4M)) and expected sampling RMSE at the same compute: sqrt(mean_m p(1-p)/N_used)."""
import json, os
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(os.path.join(H, "eval_estimates.json")))
TR = json.load(open(os.path.join(H, "../debiased_series/truth.json")))
truth = {"orig": 0.95317945}
for s in ("0.0025", "0.005"):
    for k, v in enumerate(TR[s]):
        truth[f"{s}_{k}"] = v
for grp in ("orig", "0.0025", "0.005"):
    for B in (1000, 4000, 16000, 64000):
        ks = [k for k in E if k.split("|")[1] == str(B) and k.split("|")[0].startswith(grp)]
        if not ks:
            continue
        err = np.array([E[k]["est"] - truth[k.split("|")[0]] for k in ks])
        used = np.mean([E[k]["used"] for k in ks])
        samp = np.sqrt(np.mean([truth[k.split("|")[0]] * (1 - truth[k.split("|")[0]]) for k in ks]) / used)
        tse = np.sqrt(np.mean([truth[k.split("|")[0]] * (1 - truth[k.split("|")[0]]) / 4e6 for k in ks]))
        rmse = np.sqrt(np.mean(err ** 2))
        print(f"{grp:7s} B={B:6d} used={used:8.0f} tube RMSE={100*rmse:.3f}pp  mean err={100*err.mean():+.3f}pp  "
              f"sampling RMSE={100*samp:.3f}pp  MSE gain={samp**2/rmse**2:5.2f}  (truth s.e. {100*tse:.3f}pp)  errs(pp)={np.round(100*err,3).tolist()}")
