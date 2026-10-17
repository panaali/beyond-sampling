"""Print the T75 tables from eval75.json. usage: python report75.py"""
import json, os
d = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval75.json")))
print(f"fresh noise seeds {d['noise_seeds']}, {d['n_per_seed']} copies each, sigma 0.05 (abs)")
for zs, s in d["zoo"].items():
    print(f"zoo seed {zs}: clean acc {s['clean']['truth']:.4f}, noisy mean acc {s['mean_truth']:.3f}; exact ops {s['exact_ops']:.0f} "
          f"(strict {s['exact_ops_strict']:.0f}); T75 cells/model {s['t75_cells_mean']:.0f}, resolved {s['t75_resolved_frac']:.0%} of cutting gates")
    for name in ("T75", "B", "C"):
        r = s[name]
        print(f"   {name:4s} clean err {100*r['clean_err']:+.2f}pp (ratio {r['clean_ratio']:.2f}) | noisy rms {100*r['rms']:.2f}% "
              f"bias {100*r['bias']:+.2f}% max {100*r['max_abs']:.1f}% ops {r['ops']:.0f} MSE ratio {r['mse_ratio']:.3f} "
              f"[{r['ratio_ci'][0]:.3f},{r['ratio_ci'][1]:.3f}] per noise seed {[round(x, 3) for x in r['ratio_per_noise_seed']]}")
print("done in (report)")
