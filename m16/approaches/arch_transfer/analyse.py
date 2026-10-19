"""M1 table: F per candidate and budget (T429 definition), bootstrap CI over models (rules nested), per-model F.

F = mean_rows[(est - p)^2 - se_p^2] / mean_rows[p(1-p)/cost], rows = (model, rule). Bars: strict "indicated" if point
F <= 1 and every model's rule-averaged F <= 1; lenient likewise with 10 (README section 3).
usage: python analyse.py  -> results/m1_table.md, results/m1.json
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TR = {}
for f in glob.glob(os.path.join(HERE, "truths", "*.json")):
    r = json.load(open(f)); TR[(r["tag"], r["seed"])] = r


def F_of(rows):
    e = np.array([(r["est"] - TR[(r["tag"], r["seed"])]["truth"]) ** 2 - TR[(r["tag"], r["seed"])]["se"] ** 2 for r in rows])
    v = np.array([TR[(r["tag"], r["seed"])]["truth"] * (1 - TR[(r["tag"], r["seed"])]["truth"]) / r["cost"] for r in rows])
    return e.mean() / v.mean()


def main():
    out = []
    allrows = [json.load(open(f)) for f in glob.glob(os.path.join(HERE, "runs", "m1_*.json"))]
    for tag in sorted({r["tag"] for r in allrows}):
        rows = [r for r in allrows if r["tag"] == tag]
        for B in sorted({r["budget"] for r in rows}):
            rb = [r for r in rows if r["budget"] == B and (r["tag"], r["seed"]) in TR]
            seeds = sorted({r["seed"] for r in rb})
            if not rb:
                continue
            F = F_of(rb)
            per = {s: F_of([r for r in rb if r["seed"] == s]) for s in seeds}
            rng = np.random.default_rng(0); bs = []
            for _ in range(2000):
                pick = rng.choice(seeds, len(seeds))
                bs.append(F_of([r for s in pick for r in rb if r["seed"] == s]))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            bias = np.mean([r["est"] - TR[(r["tag"], r["seed"])]["truth"] for r in rb])
            rmse = np.sqrt(np.mean([(r["est"] - TR[(r["tag"], r["seed"])]["truth"]) ** 2 for r in rb]))
            out.append({"tag": tag, "budget": B, "n_rows": len(rb), "seeds": seeds, "F": F, "ci": [lo, hi],
                        "F_per_model": per, "bias_pp": 100 * bias, "rmse_pp": 100 * rmse,
                        "cost": float(np.mean([r["cost"] for r in rb])), "mode": rb[0]["mode"], "radial": rb[0]["radial"],
                        "strict": bool(F <= 1 and max(per.values()) <= 1),
                        "lenient": bool(F <= 10 and max(per.values()) <= 10)})
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "m1.json"), "w"), indent=1)
    lines = ["| candidate | walk | radial | FPE budget | rows | F [95% CI over models] | F per model | bias pp | RMSE pp | strict | lenient |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for o in out:
        mode = "exact PL" if o["mode"] == 1 else ("exact poly" if isinstance(o["mode"], int) else "numerical")
        lines.append(f"| {o['tag']} | {mode} | {'yes' if o['radial'] else 'no'} | {o['budget']:.0e} | {o['n_rows']} | "
                     f"{o['F']:.2f} [{o['ci'][0]:.2f}, {o['ci'][1]:.2f}] | "
                     f"{', '.join(f'{v:.2f}' for v in o['F_per_model'].values())} | {o['bias_pp']:+.4f} | {o['rmse_pp']:.4f} | "
                     f"{'indicated' if o['strict'] else 'no'} | {'indicated' if o['lenient'] else 'no'} |")
    open(os.path.join(HERE, "results", "m1_table.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
