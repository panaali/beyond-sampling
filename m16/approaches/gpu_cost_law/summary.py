"""T192/T193 tables from results/runs/*.json and results/mc/*.json -> results/summary.json (+ printed table).
MSE ratio (ops) = err^2 / (p (1 - p) / N), N = ops / leaderboard.fpe_flops(H, T), err = recipe108 - exact, as E46.
MSE ratio (wall) = same with N = recipe wall seconds x MC samples/s of mc_gpu.py on the same A100 (float64)."""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from leaderboard import fpe_flops  # noqa: E402

R = {}
for f in glob.glob(os.path.join(HERE, "results", "runs", "*.json")):
    r = json.load(open(f))
    if r.get("halves_run", r.get("halves")) != r.get("halves") or r.get("acc") is None:
        continue
    R[(r["H"], r["T"], str(r["model"]).replace(":", "_"), r["mode"])] = r
AUD = {}
for f in glob.glob(os.path.join(HERE, "results", "audit", "*.json")):
    r = json.load(open(f)); AUD[(r["H"], r["T"], r["model"].replace(":", "_"), r["mode"])] = r
SAMPLE_FLOPS = {(8, 4): 682, (8, 5): 853}       # T196 ruling: complete FLOPs of one sampled input (RNG free)
AUD_FACTOR = 1.37                                # mean complete/charged over the 9 audited (8,4) runs (audit.py)
MC = {}
for f in glob.glob(os.path.join(HERE, "results", "mc", "*.json")):
    r = json.load(open(f)); MC[(r["H"], r["T"], r["model"].replace(":", "_"))] = r

rows = []
for (H, T, m, mode), ex in sorted(R.items()):
    if mode != "exact":
        continue
    rec = R.get((H, T, m, "recipe108")); mc = MC.get((H, T, m))
    p = ex["acc"]; row = dict(H=H, T=T, model=m, exact=p, exact_ops=ex["ops"], exact_cells=ex["cells"],
                              exact_simplices=ex["simplices"], exact_wall=ex["sec_wall"])
    if mc:
        row.update(mc=mc["acc"], mc_se=mc["se"], z_mc=(mc["acc"] - p) / mc["se"], mc_per_sec=mc["n"] / mc["sec"])
    if rec:
        err = rec["acc"] - p; N = rec["ops"] / fpe_flops(H, T)
        row.update(recipe=rec["acc"], err_pp=100 * err, recipe_ops=rec["ops"], recipe_cells=rec["cells"],
                   recipe_simplices=rec["simplices"], recipe_wall=rec["sec_wall"],
                   mse_ratio_ops=err ** 2 / (p * (1 - p) / N))
        a = AUD.get((H, T, m, "recipe108"))
        if (H, T) in SAMPLE_FLOPS:
            fl = a["flops_complete"] if a else AUD_FACTOR * rec["ops"]
            row.update(recipe_flops=fl, flops_measured=bool(a),
                       mse_ratio_flops=err ** 2 / (p * (1 - p) / (fl / SAMPLE_FLOPS[(H, T)])))
        if mc:
            Nw = rec["sec_wall"] * mc["n"] / mc["sec"]
            row["mse_ratio_wall"] = err ** 2 / (p * (1 - p) / Nw)
    rows.append(row)
json.dump(rows, open(os.path.join(HERE, "results", "summary.json"), "w"), indent=1)
f = lambda v, s: (s % v) if v is not None else "-"
print("| (H,T) | model | exact acc | MC z | recipe err pp | ops (exact / recipe) | cells (exact / recipe) | wall s (exact / recipe) | MSE ratio ops / T196 FLOPs / wall |")
print("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| ({r['H']},{r['T']}) | {r['model']} | {r['exact']:.9f} | {f(r.get('z_mc'), '%+.2f')} | {f(r.get('err_pp'), '%+.6f')} | "
          f"{r['exact_ops']:.3g} / {f(r.get('recipe_ops'), '%.3g')} | {r['exact_cells']:.0f} / {f(r.get('recipe_cells'), '%.0f')} | "
          f"{r['exact_wall']:.0f} / {f(r.get('recipe_wall'), '%.0f')} | {f(r.get('mse_ratio_ops'), '%.2g')} / "
          f"{f(r.get('mse_ratio_flops'), '%.2g')}{'' if r.get('flops_measured', True) else '*'} / {f(r.get('mse_ratio_wall'), '%.2g')} |")
