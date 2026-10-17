"""Print the fresh-seed tables from fresh_*.json (logs can be lost when another agent's checkout replaces files).
usage: python report.py"""
import glob, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for f in sorted(glob.glob(os.path.join(HERE, "fresh_*.json"))):
    d = json.load(open(f)); print(f"{os.path.basename(f)}: sigma {d['sigma']}, noise seeds {d['seeds']}, {d['n']} copies each")
    for zs, r in d["zoo"].items():
        print(f"  zoo {zs}: acc {r['mean_truth']:.3f} rms {100*r['rms']:.2f}% bias {100*r['bias']:+.3f}% max {100*r['max_abs']:.1f}% "
              f"ops {r['ops']:.0f} ratio {r['mse_ratio']:.4f} [{r['ci'][0]:.4f},{r['ci'][1]:.4f}] per seed {[round(x, 4) for x in r['per_seed']]} "
              f"reimpl maxdiff {r['reimpl_maxdiff']:.1e}")
sw = os.path.join(HERE, "sweep.json")
if os.path.exists(sw):
    print("sweep.json: MSE ratio per zoo seed 0-4 (ops in k, recipe.py charge)")
    for r in json.load(open(sw)):
        print(f"  {r['name']:42s} " + " || ".join(f"{s.split(' (')[0]}: " + " ".join(f"{x[0]:.4f}" for x in r[s]) + f" (max {max(x[0] for x in r[s]):.3f}; ops "
              + " ".join(f"{x[1]/1000:.1f}" for x in r[s]) + ")" for s in ("dev seed 0 (n=32)", "fresh seed 30 (n=128)")))
