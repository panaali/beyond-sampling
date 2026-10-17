"""T195 phase 2 table from prune/*.json. Cost convention (T196): estimator FLOPs = 2 x m43 ops, N = FLOPs / fpe_flops.
SD = sqrt(p(1-p)/N) at the retained cost; ratio_act = (acc - exact)^2 / SD^2; ratio_B = B^2 / SD^2 (pruning bound only;
recipe108's own error is not certified). usage: python prune_report.py > prune_report.md"""
import glob, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
EX = {3: {int(k): v for k, v in json.load(open(os.path.join(HERE, "ref", "exact_H8_T3.json"))).items()}}
sm = json.load(open(os.path.join(HERE, "..", "estT_fix", "summary.json")))
EX[4] = {s: sm[f"zoo{s}"]["exact"] for s in range(5)}
EX[5] = {0: json.load(open(os.path.join(HERE, "..", "estT_fix", "run5", "zoo0_exact.json")))["acc"]}
REF = {}
for f in glob.glob(os.path.join(HERE, "ref", "H8_T*_s*.json")):
    b = os.path.basename(f)[:-5]; REF[(int(b[4]), int(b[-1]))] = json.load(open(f))
rows = [json.load(open(f)) for f in glob.glob(os.path.join(HERE, "prune", "*.json"))]
rows.sort(key=lambda r: (r["T"], r["test"], -r["tau"], r["seed"]))
print("| T | test | tau | seed | retained cells (recipe) | FLOPs (recipe) | pruned | err vs exact | bound B | B oracle (mixed only) | SD | ratio act | ratio B |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
agg = {}
for r in rows:
    if r["tau"] == 0:
        continue
    T, s = r["T"], r["seed"]; p = EX[T][s]; sd = (p * (1 - p) / r["N"]) ** .5; e = r["acc"] - p; ref = REF[(T, s)]
    ra, rb = e * e / sd / sd, r["bound"] ** 2 / sd / sd
    bo = r.get("bound_oracle_uncertain")
    print(f"| {T} | {r['test']} | {r['tau']:g} | {s} | {r['retained_cells']} ({ref['cells']}) | {r['flops']:.3g} ({2 * ref['ops']:.3g}) | {r['pruned']} | "
          f"{e:+.2e} | {r['bound']:.2e} | {'' if bo is None else f'{bo:.2e}'} | {sd:.2e} | {ra:.3g} | {rb:.3g} |")
    a = agg.setdefault((r["test"], r["tau"], T), []); a.append((r["retained_cells"], ref["cells"], rb, ra, r["flops"]))
print("\n## Pooled: mean retained cells per model and growth along T at fixed tau\n")
print("| test | tau | T=3 cells | T=4 cells | T=5 cells | x(3->4) | x(4->5) | max ratio_B per T | max ratio_act per T |")
print("|---|---|---|---|---|---|---|---|---|")
for test in ("cap", "mass"):
    for tau in sorted({k[1] for k in agg if k[0] == test}, reverse=True):
        c = {T: np.mean([x[0] for x in agg[(test, tau, T)]]) for T in (3, 4, 5) if (test, tau, T) in agg}
        mb = {T: max(x[2] for x in agg[(test, tau, T)]) for T in c}; ma = {T: max(x[3] for x in agg[(test, tau, T)]) for T in c}
        g34 = c[4] / c[3] if 3 in c and 4 in c else float("nan"); g45 = c[5] / c[4] if 4 in c and 5 in c else float("nan")
        print(f"| {test} | {tau:g} | {c.get(3, float('nan')):.0f} | {c.get(4, float('nan')):.0f} | {c.get(5, float('nan')):.0f} | {g34:.1f} | {g45:.1f} | "
              + ", ".join(f"T{T}:{v:.3g}" for T, v in mb.items()) + " | " + ", ".join(f"T{T}:{v:.3g}" for T, v in ma.items()) + " |")
rc = {T: np.mean([REF[(T, s)]["cells"] for s in range(5) if (T, s) in REF]) for T in (3, 4, 5)}
print(f"\nunpruned recipe108 mean cells: T3 {rc[3]:.0f}, T4 {rc[4]:.0f}, T5 {rc[5]:.0f} (seed 0 only at T5)")
