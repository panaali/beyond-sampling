"""T442 analysis (fixed before scoring). usage: python analyse.py -> analysis.json + analysis.log (stdout).
F(B) = mean_m[(est-p)^2 - se_truth^2] / mean_m[p(1-p)/cost]  (T429 analyse.py definition, DEV3 1e8 truths).
CIs: cluster bootstrap over the 100 models (2000 resamples, rng 0), the same model indices for every config and
budget, so family comparisons and exponents are paired."""
import json
import os

import numpy as np

from configs import BUDGETS, CONFIGS, FROZEN, PRIMARY

HERE = os.path.dirname(os.path.abspath(__file__))
tr = json.load(open(os.path.join(HERE, "..", "..", "..", "data", "dev3", "truths.json")))["truths"]
names = sorted(tr)
p = np.array([tr[n]["acc"] for n in names]); se = np.array([tr[n]["se"] for n in names])
idx = np.random.default_rng(0).integers(0, len(p), (2000, len(p)))


def path(fam, seed, b):
    if (fam, seed) == FROZEN:
        return os.path.join(HERE, "..", "walk_budget", "runs", f"t10_{b}.jsonl")
    return os.path.join(HERE, "runs", f"{fam}_{seed}_{b}.jsonl")


# per config/budget: numerator rows (se2) and denominator rows (v), shape (100,)
D = {}
for fam, seed in [FROZEN] + CONFIGS:
    for b in BUDGETS:
        f = path(fam, seed, b)
        if not os.path.exists(f):
            continue
        r = {x["model"]: x for x in map(json.loads, open(f))}
        if len(r) < 100:
            continue
        e = np.array([r[n]["est"] for n in names]); c = np.array([r[n]["cost"] for n in names])
        D[fam, seed, b] = ((e - p) ** 2 - se ** 2, p * (1 - p) / c, c.mean(), 100 * (e - p).mean())


def F(k, ii=slice(None)):
    return D[k][0][ii].mean() / D[k][1][ii].mean()


def Fboot(k):
    return np.array([F(k, ii) for ii in idx])


def ci(a):
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


fams = {"sobol": [FROZEN] + [c for c in CONFIGS if c[0] == "sobol"]}
for fam in ("lat", "latb", "lat0", "halton"):
    fams[fam] = [c for c in CONFIGS if c[0] == fam]
out = dict(configs={}, families={}, frozen={}, compare={})
for fam, seed in [FROZEN] + CONFIGS:
    row = {}
    for b in BUDGETS:
        k = (fam, seed, b)
        if k in D:
            bt = Fboot(k); lo, hi = ci(bt)
            row[b] = dict(F=F(k), lo=lo, hi=hi, cost=D[k][2], bias_pp=D[k][3], strict=hi < 1, lenient=hi < 10)
    bl = [b for b in BUDGETS if (fam, seed, b) in D]
    if len(bl) >= 2:
        x = np.log([D[fam, seed, b][2] for b in bl])
        bb = [np.polyfit(x, np.log([max(F((fam, seed, b), ii), 1e-9) for b in bl]), 1)[0] for ii in idx]
        row["beta"] = dict(budgets=bl, beta=float(np.polyfit(x, np.log([max(F((fam, seed, b)), 1e-9) for b in bl]),
                                                             1)[0]), ci=ci(bb))
    out["configs"][f"{fam}_{seed}"] = row
    s = " ".join(f"{b:.0e}: {row[b]['F']:.3f} [{row[b]['lo']:.3f}, {row[b]['hi']:.3f}]{'*' if row[b]['strict'] else ''}"
                 for b in BUDGETS if b in row)
    print(f"{fam:6s} {seed:>10}  {s}  beta {row['beta']['beta']:+.3f} [{row['beta']['ci'][0]:+.3f}, "
          f"{row['beta']['ci'][1]:+.3f}]" if "beta" in row else f"{fam:6s} {seed:>10}  {s}", flush=True)
print("(* = strict bar cleared: CI upper < 1; lenient F <= 10 cleared wherever CI upper < 10)")

# family summaries: mean F over the family's seeds (expected F for a random rule of the family), seed spread
for fam, cl in fams.items():
    res = {}
    for b in BUDGETS:
        ks = [(f_, s_, b) for f_, s_ in cl if (f_, s_, b) in D]
        if not ks:
            continue
        Fs = np.array([F(k) for k in ks])
        bt = np.array([np.mean([F(k, ii) for k in ks]) for ii in idx])
        res[b] = dict(n=len(ks), meanF=float(Fs.mean()), ci=ci(bt), sd=float(Fs.std(ddof=1)) if len(ks) > 1 else None,
                      min=float(Fs.min()), max=float(Fs.max()), n_strict=int(sum(out["configs"][f"{k[0]}_{k[1]}"][b]
                                                                              ["strict"] for k in ks)))
        r = res[b]
        print(f"family {fam:6s} {b:.0e}: n {r['n']}  mean F {r['meanF']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]  "
              f"sd {r['sd'] if r['sd'] is None else round(r['sd'], 3)}  range [{r['min']:.3f}, {r['max']:.3f}]  "
              f"strict {r['n_strict']}/{r['n']}", flush=True)
    bl = [b for b in BUDGETS if b in res and res[b]["n"] == len(cl)]
    if len(bl) >= 2:
        betas = [out["configs"][f"{f_}_{s_}"]["beta"]["beta"] for f_, s_ in cl]
        res["beta_mean"] = float(np.mean(betas)); res["beta_sd"] = float(np.std(betas, ddof=1)) if len(betas) > 1 else None
        print(f"family {fam:6s} exponent: mean {res['beta_mean']:+.3f}  sd across seeds {res['beta_sd']}")
    out["families"][fam] = res

# frozen seed typicality: rank among the 11 Sobol seeds, and z vs the 10 new seeds
for b in PRIMARY:
    ks = [(f_, s_, b) for f_, s_ in fams["sobol"] if (f_, s_, b) in D]
    if (FROZEN + (b,)) not in D or len(ks) < 3:
        continue
    Fs = np.array([F(k) for k in ks]); f0 = F(FROZEN + (b,)); new = Fs[1:]
    rank = int((Fs < f0).sum() + 1)
    out["frozen"][b] = dict(F=f0, rank=rank, of=len(Fs), new_mean=float(new.mean()), new_sd=float(new.std(ddof=1)),
                            z=float((f0 - new.mean()) / new.std(ddof=1)), typical=1 < rank < len(Fs))
    r = out["frozen"][b]
    print(f"frozen 31415 {b:.0e}: F {f0:.3f}, rank {rank}/{len(Fs)} (1 = best), new seeds {r['new_mean']:.3f} +- "
          f"{r['new_sd']:.3f}, z {r['z']:+.2f} -> {'TYPICAL' if r['typical'] else 'EXTREME (best or worst)'}")

# family vs Sobol: paired bootstrap ratio of family-mean F; reliably better iff CI upper < 1 at both primary budgets
for fam in ("lat", "latb", "lat0", "halton"):
    res = {}
    for b in PRIMARY:
        ka = [(f_, s_, b) for f_, s_ in fams[fam] if (f_, s_, b) in D]
        ks = [(f_, s_, b) for f_, s_ in fams["sobol"] if (f_, s_, b) in D]
        if len(ka) < len(fams[fam]) or len(ks) < 2:
            continue
        rat = np.array([np.mean([F(k, ii) for k in ka]) / np.mean([F(k, ii) for k in ks]) for ii in idx])
        res[b] = dict(ratio=float(np.mean([F(k) for k in ka]) / np.mean([F(k) for k in ks])), ci=ci(rat))
        print(f"{fam:6s}/sobol {b:.0e}: ratio of mean F {res[b]['ratio']:.3f} [{res[b]['ci'][0]:.3f}, {res[b]['ci'][1]:.3f}]")
    if len(res) == len(PRIMARY):
        res["reading"] = ("BETTER" if all(res[b]["ci"][1] < 1 for b in PRIMARY) else
                          "WORSE" if all(res[b]["ci"][0] > 1 for b in PRIMARY) else "NOT DISTINGUISHED")
        print(f"{fam:6s} vs sobol: {res['reading']}")
    out["compare"][fam] = res
json.dump(out, open(os.path.join(HERE, "analysis.json"), "w"), indent=1, default=float)
