"""T376 tables and the pre-registered verdict from results/cond_k<k>_*.json (paired with T3's results/<model>.json)."""
import json, os
import numpy as np
order = ["orig"] + [f"seed{i}" for i in range(5)] + [f"s{s}_{i}" for s in (0.0025, 0.005) for i in range(6)]
V = ("D3", "D5", "R3", "X3", "skel", "D3u")
for k in (4, 5):
    R = {n: json.load(open(f"results/cond_k{k}_{n}.json")) for n in order if os.path.exists(f"results/cond_k{k}_{n}.json")}
    if not R: continue
    T3 = {n: json.load(open(f"results/{n}.json")) for n in R}
    ns = list(R)
    print(f"\n### T376 k={k}: bias pp (s.e.); T3 columns for reference\n| model | T3 none V1 | T3 D3 V1 | " + " | ".join(V)
          + " | L0/L1/L2/L3 share (D3) | cyl. violation T3-V1 / D3 / R3 |\n|---|---|---|" + "---|" * len(V) + "---|---|")
    for n in ns:
        b, se, t = R[n]["bias_pp"], R[n]["se_pp"], T3[n]["bias_pp"]
        print(f"| {n} | {t[f'none_k{k}_V1']:+.3f} | {t[f'D3_k{k}_V1']:+.3f} | " + " | ".join(f"{b[v]:+.3f} ({se[v]:.3f})" for v in V)
              + " | " + "/".join(f"{a:.2f}" for a in R[n]["level_frac"]["D3"]) + " | "
              + " / ".join(f"{R[n]['cyl_violation'][v]:.3f}" for v in ("T3_V1", "D3", "R3")) + " |")
    print("| mean | | | " + " | ".join(f"{np.mean([R[n]['bias_pp'][v] for n in ns]):+.3f}" for v in V) + " | | |")
    print("| mean abs | | | " + " | ".join(f"{np.mean([abs(R[n]['bias_pp'][v]) for n in ns]):.3f}" for v in V) + " | | |")
    print("| max abs | | | " + " | ".join(f"{np.max([abs(R[n]['bias_pp'][v]) for n in ns]):.3f}" for v in V) + " | | |")
    b = np.array([R[n]["bias_pp"]["D3"] for n in ns])
    g = lambda a, c: sum(abs(R[n]["bias_pp"][a]) < abs(R[n]["bias_pp"][c]) - 2 * np.hypot(R[n]["se_pp"][a], R[n]["se_pp"][c]) for n in ns)
    print(f"\nk={k} over {len(ns)} models: D3 max|bias| {np.abs(b).max():.3f}, within .08: {(np.abs(b) <= .08).sum()}, "
          f"beats X3: {g('D3', 'X3')}; clocks vs random: D3 beats R3 on {g('D3', 'R3')}, R3 beats D3 on {g('R3', 'D3')}")
    if k == 4:
        print("PRE-REGISTERED VERDICT (k=4):", "PASS" if len(ns) == 18 and np.all(np.abs(b) <= .08) and g("D3", "X3") >= 12 else "FAIL")
