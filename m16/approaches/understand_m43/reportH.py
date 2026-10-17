"""Tables for T112 from evalH_<H>.json (H = 4, 5, 6, 8; T = 3). usage: python reportH.py"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
print("| rung | zoo seed | clean acc | clean err | noisy acc | noisy RMSE | max | ops (exact) | MSE ratio [CI] | regions | exact leaves | recipe cells | resolved | rescale change |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for H in (4, 5, 6, 8):
    d = json.load(open(os.path.join(HERE, f"evalH_{H}.json")))
    for k, s in sorted(d["zoo"].items()):
        print(f"| ({H},3) | {k} | {s['clean_truth']:.4f} | {100*s['clean_err']:+.4f} pp | {s['mean_truth']:.3f} | {100*s['rms']:.3f}% | {100*s['max_abs']:.2f}% | "
              f"{s['ops']/1000:.0f}k ({s['exact_ops']/1000:.0f}k) | {s['ratio']:.4f} [{s['ci'][0]:.4f}, {s['ci'][1]:.4f}] | {s['regions']} | {s['clean_leaves']} | "
              f"{s['clean_cells']} | {s['resolved_frac']:.0%} | {s['max_rescale_change']:.0e} |")
