"""Renders the T431 results section of README.md from results/hp5_t431.json (group aggregates only)."""
import json
d = json.load(open("results/hp5_t431.json"))
f = lambda v, k="F": f"{v[k]:.3f} [{v[k+'_ci'][0]:.3f}, {v[k+'_ci'][1]:.3f}]"
L = ["### T431 results (HP5, 100 sealed retrains; `python3 t431_analyse.py` → `results/hp5_t431.json`)",
     "| run | C mean (FPE) | cost ≥ B/2 | **F** (truth var subtracted) | F raw | T281 ratio | bias pp | RMS pp | truth share | strict (CI upper < 1) | lenient (CI upper < 10) |",
     "|---|---|---|---|---|---|---|---|---|---|---|"]
for k, v in d.items():
    if "FLOPs/5948" in k:
        continue
    L.append(f"| {k} | {v['C_mean_fpe']:,.0f} | {'yes' if v['cost_ok'] else 'NO'} | {f(v)} | {f(v,'F_raw')} | {v['R281']:.3f} | {v['bias_pp']:+.4f} | {v['rms_pp']:.4f} | {v['truth_share']:.3f} | {'PASS' if v['strict_ci'] else 'fail'} | {'PASS' if v['lenient_ci'] else 'fail'} |")
L += ["", "Sensitivity: the recipes' FPE unit is T(2H²+2H) = 5,440 FLOPs; charging C as FLOPs/5,948 (×0.915) instead:"]
for k, v in d.items():
    if "FLOPs/5948" in k:
        L.append(f"- {k.replace(' [C in FLOPs/5948]', '')}: F {f(v)}")
open("results/hp5_t431_table.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
