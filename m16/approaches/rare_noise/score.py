"""T371 scoring (pre-registered): est/truth per (model, t0); go line >= 80% of the NOISY points within [1/2, 2].
Points whose truth has < 30 hits are listed but excluded from the go count (truth-limited).
usage: python score.py -> results/score.json and a table on stdout"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from models import BASES, NCOPY, SIGMAS  # noqa: E402

est = {}
for fn in sorted(glob.glob(os.path.join(HERE, "results", "est_*.json"))):
    est.update(json.load(open(fn)))
tr = {json.load(open(fn))["t0"]: json.load(open(fn)) for fn in glob.glob(os.path.join(HERE, "results", "truth_*.json"))}
T0S = sorted(tr); BOUND = 7.037785093518071e-08          # P(g23 > 3.5), the T_hi cut's declared bound
rows = []
for nm in sorted(est):
    base, sig, cp = (nm.split("@")[0], float(nm.split("@")[1].split("#")[0]), int(nm.split("#")[1])) if "@" in nm else (nm, 0.0, -1)
    for t0 in T0S:
        T = tr[t0]; k = T["hits"][nm]; n = T["n"]; q = k / n; p = T["p_gap"] * q
        se = T["p_gap"] * np.sqrt(max(q * (1 - q), 1e-300) / n); e = est[nm]["est"][str(t0)]
        rows.append(dict(model=nm, base=base, sigma=sig, copy=cp, t0=t0, est=e, truth=p, se=se, hits=k,
                         ratio=e / p if p > 0 else None, far_share=T["hits_g23_gt_3p5"][nm] / k if k else None,
                         tie_share=T["hits_g12_lt_p1"][nm] / k if k else None, C_fpe=est[nm]["C_fpe"], n=n))


def summ(rs):
    ok = [r for r in rs if r["hits"] >= 30]; rat = np.array([r["ratio"] for r in ok])
    if not len(rat):
        return dict(n=0)
    return dict(n=len(ok), within2=int(((rat >= .5) & (rat <= 2)).sum()), frac2=float(((rat >= .5) & (rat <= 2)).mean()),
                within3=int(((rat >= 1 / 3) & (rat <= 3)).sum()), median=float(np.median(rat)),
                mean_log10=float(np.log10(np.maximum(rat, 1e-300)).mean()), n_high=int((rat > 1).sum()), n_low=int((rat < 1).sum()),
                median_abs_log10=float(np.median(np.abs(np.log10(np.maximum(rat, 1e-300))))), excluded=len(rs) - len(ok))


noisy = [r for r in rows if r["sigma"] > 0]
out = dict(rows=rows, go_line=0.8, all_noisy=summ(noisy), clean=summ([r for r in rows if r["sigma"] == 0]),
           by_sigma={str(s): summ([r for r in noisy if r["sigma"] == s]) for s in SIGMAS},
           by_base_sigma={f"{b}@{s:g}": summ([r for r in noisy if r["base"] == b and r["sigma"] == s]) for b in BASES for s in SIGMAS},
           by_t0={str(t): summ([r for r in noisy if r["t0"] == t]) for t in T0S},
           in_HR1_range=summ([r for r in noisy if 2e-7 <= r["truth"] <= 2e-4]))
out["verdict"] = "GO" if out["all_noisy"].get("frac2", 0) >= 0.8 else "NO GO"
json.dump(out, open(os.path.join(HERE, "results", "score.json"), "w"), indent=1)
print(f"{'model':22s} {'t0':>5s} {'truth':>10s} {'rel.se':>7s} {'est':>10s} {'ratio':>7s} {'tie%':>5s} {'>3.5%':>6s}")
for r in rows:
    print(f"{r['model']:22s} {r['t0']:5.2f} {r['truth']:10.3e} {r['se'] / r['truth'] if r['truth'] else 0:7.3f} {r['est']:10.3e} "
          f"{r['ratio'] if r['ratio'] is not None else float('nan'):7.3f} {100 * (r['tie_share'] or 0):5.1f} {100 * (r['far_share'] or 0):6.2f}")
for k in ("clean", "all_noisy", "in_HR1_range"):
    print(k, out[k])
for k, v in out["by_sigma"].items():
    print("sigma", k, v)
for k, v in out["by_base_sigma"].items():
    print("  ", k, v)
for k, v in out["by_t0"].items():
    print("t0", k, v)
print("VERDICT", out["verdict"], "(declared T_hi bound per point: %.2e)" % BOUND)
