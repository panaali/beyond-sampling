"""T234 grid3 analysis: per config (name prefix before _s<seed>) n, mean, sd, median, top-5 mean, max, bootstrap 95% CI
of the mean (2000 resamples, numpy seed 0), share of models above M_16,10's .9531175, and the error anatomy means
(blind / 2-3 / 1-2 / other error rates). Source: results/grid3.json (in-kernel float32 1e8-sample eval, independent
Philox stream per model). usage: python grid3_analysis.py -> results/grid3_summary.json"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); T = 0.9531175
ev = json.load(open(os.path.join(HERE, "results", "grid3.json")))["eval"]; g = {}
for nm, e in ev.items():
    g.setdefault(nm.rsplit("_s", 1)[0], []).append(e)
rng = np.random.default_rng(0); out = {}
for k in sorted(g):
    a = np.array([e["acc"] for e in g[k]]); bm = [rng.choice(a, len(a)).mean() for _ in range(2000)]
    out[k] = dict(n=len(a), mean=a.mean(), sd=a.std(ddof=1), median=float(np.median(a)), top5=float(np.sort(a)[-5:].mean()),
                  max=a.max(), ci95=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))], above_target=float((a > T).mean()),
                  **{f"mean_{x}": float(np.mean([e[x] for e in g[k]])) for x in ("err_blind", "err_23", "err_12", "err_other")})
    o = out[k]; print("%-14s n %3d mean %.4f [%.4f, %.4f] sd %.4f med %.4f top5 %.4f max %.4f >T %.2f | blind %.4f 2-3 %.4f 1-2 %.4f other %.4f" % (
        k, o["n"], o["mean"], *o["ci95"], o["sd"], o["median"], o["top5"], o["max"], o["above_target"], o["mean_err_blind"], o["mean_err_23"], o["mean_err_12"], o["mean_err_other"]))
json.dump({k: {a: (float(b) if isinstance(b, (np.floating, float)) else b) for a, b in v.items()} for k, v in out.items()},
          open(os.path.join(HERE, "results", "grid3_summary.json"), "w"), indent=1)
