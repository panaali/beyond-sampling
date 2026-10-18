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


sys_path = os.path.join(ROOT, "m16"); import sys; sys.path.insert(0, sys_path)
from leaderboard import flops
ANAT = os.path.join(ROOT, "m16", "approaches", "width16_probe", "anatomy")


def pq(model):
    """(p, samples per unit cost). Costs are FPE / model evals (1 sample each), except enum (estT ops at (H,4):
    T196 FLOPs = 2 ops; one sample = c_s(H,4) FLOPs)."""
    if model.startswith("zoo"):
        H = int(model[3:].split("_")[0]); acc = json.load(open(os.path.join(ANAT, f"H{H}_T4_zoo0_chamber.json")))["acc"]
        return acc, 2.0 / (flops(H, 4) + 4 * H + 3 * 4 - 2)
    return (0.9531174789 if model == "orig" else TR[model]["acc"]), 1.0


def ranges(K, a):
    """ratio(C) = K C^(1 - 2a); budgets where ratio <= 10 and <= 1 (C in the family's cost unit)."""
    out = {}
    for bar in (10.0, 1.0):
        if abs(1 - 2 * a) < 1e-9:
            out[str(bar)] = "all" if K <= bar else "none"
        elif a > 0.5:
            out[str(bar)] = f"C >= {(K / bar) ** (1 / (2 * a - 1)):.3g}"
        else:
            out[str(bar)] = f"C <= {(bar / K) ** (1 / (1 - 2 * a)):.3g}"
    return out


out = {}
for (fam, model), byk in sorted(G.items()):
    ks = sorted(byk); C = [np.mean([r["cost"] for r in byk[k]]) for k in ks]
    bias = [np.mean([r["err"] for r in byk[k]]) for k in ks]; rmse = [np.sqrt(np.mean([r["err"] ** 2 for r in byk[k]])) for k in ks]
    floor = 2 * se_truth(model); ok = [abs(b) > floor for b in bias]
    o = dict(knobs=ks, cost=C, bias=bias, rmse=rmse, span=C[-1] / C[0], reps=len(byk[ks[0]]), floor=floor,
             fit_bias=slope(C, bias), fit_bias_resolved=slope([c for c, k in zip(C, ok) if k], [b for b, k in zip(bias, ok) if k]))
    if o["reps"] > 1:
        o["fit_rmse"] = slope(C, rmse)
    p, spu = pq(model); v = p * (1 - p)
    o["ratio_measured"] = [float(r ** 2 * c * spu / v) for r, c in zip(rmse, C)]
    law = o.get("fit_rmse") or o["fit_bias"]; ys = rmse if o.get("fit_rmse") else bias
    if law:
        a = law["alpha"]; lc = np.log(C); ly = np.log(np.abs(ys)); A = float(np.exp(np.mean(ly + a * lc)))   # y = A C^-a
        K = A ** 2 * spu / v; o["predicted_ranges"] = ranges(K, a); o["law"] = dict(A=A, alpha=a, K=K)
    out[f"{fam}|{model}"] = o
    fb = o["fit_bias"]; fr = o.get("fit_rmse")
    print(f"{fam:9s} {model:9s} span x{o['span']:.0f} bias(pp) " + " ".join(f"{100*b:+.4f}" for b in bias)
          + (f" | alpha_bias {fb['alpha']:.2f} [{fb['ci'][0]:.2f},{fb['ci'][1]:.2f}]" if fb else "")
          + (f" | alpha_rmse {fr['alpha']:.2f} [{fr['ci'][0]:.2f},{fr['ci'][1]:.2f}]" if fr else "")
          + f" | resolved {sum(ok)}/{len(ok)}" + " | ratio " + " ".join(f"{x:.3g}" for x in o["ratio_measured"])
          + (f" | within10x {o['predicted_ranges']['10.0']}; <=1 {o['predicted_ranges']['1.0']}" if "predicted_ranges" in o else ""))
json.dump(out, open(os.path.join(HERE, "fit.json"), "w"), indent=1)
