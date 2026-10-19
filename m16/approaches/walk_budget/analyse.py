"""T429 analysis (fixed before scoring). usage: python analyse.py  -> analysis.json + stdout table.
F(B) = mean_m[(est-p)^2 - se_truth^2] / mean_m[p(1-p)/cost]   (score_dev3.py definition, T389/T391)
95% CI: cluster bootstrap over the 100 models (2000 resamples, rng 0), the same model indices at every budget.
Exponent beta: least-squares slope of log F on log(mean cost) over the budgets, per bootstrap resample."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
tr = json.load(open(os.path.join(HERE, "..", "..", "..", "data", "dev3", "truths.json")))["truths"]
BUDGETS = [16000, 64000, 256000, 1000000, 4000000, 10000000]
NEW = [256000, 1000000, 4000000, 10000000]
out = {}
for recipe in ("isg", "t10"):
    rows = {}
    for b in BUDGETS:
        f = os.path.join(HERE, "runs", f"{recipe}_{b}.jsonl")
        if os.path.exists(f):
            rows[b] = {r["model"]: r for r in map(json.loads, open(f))}
    bs_ = [b for b in BUDGETS if b in rows and len(rows[b]) == 100]
    if not bs_:
        continue
    names = sorted(rows[bs_[0]])
    p = np.array([tr[n]["acc"] for n in names]); se = np.array([tr[n]["se"] for n in names])
    E = {b: np.array([rows[b][n]["est"] for n in names]) for b in bs_}
    Cc = {b: np.array([rows[b][n]["cost"] for n in names]) for b in bs_}
    se2 = {b: (E[b] - p) ** 2 - se ** 2 for b in bs_}
    v = {b: p * (1 - p) / Cc[b] for b in bs_}
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(p), (2000, len(p)))

    def F(b, ii=slice(None)):
        return se2[b][ii].mean() / v[b][ii].mean()

    def beta(bl, ii=slice(None)):
        x = np.log([Cc[b].mean() for b in bl]); y = np.log([max(F(b, ii), 1e-9) for b in bl])
        return np.polyfit(x, y, 1)[0]

    res = {}
    for b in bs_:
        boot = np.array([F(b, ii) for ii in idx])
        res[b] = dict(F=F(b), lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5), cost=Cc[b].mean(),
                      bias_pp=100 * (E[b] - p).mean(), rmse_pp=100 * np.sqrt(max(se2[b].mean(), 0)),
                      truth_share=(se ** 2).mean() / ((E[b] - p) ** 2).mean(),
                      F_raw=((E[b] - p) ** 2).mean() / v[b].mean(),
                      cpu=np.mean([rows[b][n]["cpu"] for n in names]))
        r = res[b]
        print(f"{recipe} {b:>9}: cost {r['cost']:.0f}  F {r['F']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]  strict(F<=1) "
              f"{'PASS' if r['hi'] < 1 else 'not CI<1'}  lenient(F<=10) {'PASS' if r['hi'] < 10 else 'not CI<10'}  "
              f"bias {r['bias_pp']:+.3f} pp  RMSE {r['rmse_pp']:.4f} pp  truth-var share {r['truth_share']:.2f}  "
              f"F_raw {r['F_raw']:.3f}  cpu {r['cpu']:.1f}s", flush=True)
    fits = {}
    for lab, bl in (("all", bs_), ("new", [b for b in NEW if b in bs_])):
        if len(bl) >= 2:
            bb = np.array([beta(bl, ii) for ii in idx])
            fits[lab] = dict(budgets=bl, beta=beta(bl), lo=np.percentile(bb, 2.5), hi=np.percentile(bb, 97.5))
            f_ = fits[lab]
            read = "FALLING" if f_["hi"] < 0 else "RISING" if f_["lo"] > 0 else "FLAT (CI contains 0)"
            f_["reading"] = read
            print(f"{recipe} beta[{lab}: {bl[0]:.0e}-{bl[-1]:.0e}] = {f_['beta']:+.3f} [{f_['lo']:+.3f}, {f_['hi']:+.3f}]"
                  f"  per 4x budget x{4 ** f_['beta']:.2f}  -> {read}", flush=True)
    out[recipe] = dict(rows={str(b): res[b] for b in res}, fits=fits)
json.dump(out, open(os.path.join(HERE, "analysis.json"), "w"), indent=1, default=float)
