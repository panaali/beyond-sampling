"""Print the README tables from results.json (no computation beyond arithmetic). usage: python report.py"""
import json, os
import numpy as np
R = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")))
print("| tier | RMSE | bias | max abs err | ops (mean) | sampling RMSE at equal ops | MSE ratio |")
print("|---|---|---|---|---|---|---|")
for k, r in R["tiers"].items():
    print(f"| {k} | {100*r['rms']:.2f}% | {100*r['bias']:+.2f}% | {100*r['max_abs']:.1f}% | {r['ops_mean']:,.0f} | "
          f"{100*r['sampling_rmse_equal_ops']:.2f}% | {r['mse_ratio']:.3f} |")
print("\nzoo seeds: error (est - exact) and per-model MSE ratio (err^2 / (p(1-p)/(ops/72)))")
print("| tier | " + " | ".join(f"seed {s}" for s in range(5)) + " |")
print("|---|" + "---|" * 5)
for k, r in R["tiers"].items():
    cells = []
    for s in range(5):
        z = r["zoo"][str(s)]; p = z["truth"]; mse_s = p * (1 - p) / (z["ops"] / 72)
        cells.append(f"{100*z['err']:+.2f}% ({z['err']**2/mse_s:.2f})")
    print(f"| {k} | " + " | ".join(cells) + " |")
print("\nexact truths of zoo seeds:", {s: round(R['tiers'][next(iter(R['tiers']))]['zoo'][str(s)]['truth'], 5) for s in range(5)})
