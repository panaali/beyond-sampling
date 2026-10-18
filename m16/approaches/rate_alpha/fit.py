"""T271: fit alpha in |bias| ~ C^-alpha (and RMSE over rule scrambles where reps > 1) per family and model, from rows.jsonl.
OLS of log|y| on log C; CI = slope +- t_{n-2,.975} SE. Points with |bias| below 2x the truth SE (noisy copies: 3.2e7-sample
truths; target: 3.9e-7) are flagged as unresolved and the fit is repeated without them. -> fit.json"""
import json, os
from collections import defaultdict
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TR = json.load(open(os.path.join(ROOT, "results", "leaderboard_truths.json")))
se_truth = lambda m: 3.9e-7 if m == "orig" else (TR[m]["se"] if m in TR else 0.0)
rows = [json.loads(l) for l in open(os.path.join(HERE, "rows.jsonl"))]
G = defaultdict(lambda: defaultdict(list))
for r in rows:
    G[(r["fam"], r["model"])][r["knob"]].append(r)


def slope(C, y):
    C, y = np.log(np.asarray(C)), np.log(np.abs(np.asarray(y)))
    if len(C) < 3:
        return None
    res = stats.linregress(C, y); t = stats.t.ppf(0.975, len(C) - 2)
    return dict(alpha=-res.slope, ci=[-res.slope - t * res.stderr, -res.slope + t * res.stderr], n=len(C))


out = {}
for (fam, model), byk in sorted(G.items()):
    ks = sorted(byk); C = [np.mean([r["cost"] for r in byk[k]]) for k in ks]
    bias = [np.mean([r["err"] for r in byk[k]]) for k in ks]; rmse = [np.sqrt(np.mean([r["err"] ** 2 for r in byk[k]])) for k in ks]
    floor = 2 * se_truth(model); ok = [abs(b) > floor for b in bias]
    o = dict(knobs=ks, cost=C, bias=bias, rmse=rmse, span=C[-1] / C[0], reps=len(byk[ks[0]]), floor=floor,
             fit_bias=slope(C, bias), fit_bias_resolved=slope([c for c, k in zip(C, ok) if k], [b for b, k in zip(bias, ok) if k]))
    if o["reps"] > 1:
        o["fit_rmse"] = slope(C, rmse)
    out[f"{fam}|{model}"] = o
    fb = o["fit_bias"]; fr = o.get("fit_rmse")
    print(f"{fam:9s} {model:9s} span x{o['span']:.0f} bias(pp) " + " ".join(f"{100*b:+.4f}" for b in bias)
          + (f" | alpha_bias {fb['alpha']:.2f} [{fb['ci'][0]:.2f},{fb['ci'][1]:.2f}]" if fb else "")
          + (f" | alpha_rmse {fr['alpha']:.2f} [{fr['ci'][0]:.2f},{fr['ci'][1]:.2f}]" if fr else "")
          + f" | resolved {sum(ok)}/{len(ok)}")
json.dump(out, open(os.path.join(HERE, "fit.json"), "w"), indent=1)
