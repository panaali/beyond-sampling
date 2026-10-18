"""Tables and the pre-registered verdict from results/*.json."""
import json, glob
import numpy as np
from models import all_models
R = {}
for f in sorted(glob.glob("results/*.json")):
    d = json.load(open(f)); R[d["model"]] = d
order = ["orig"] + [f"seed{i}" for i in range(5)] + [f"s{s}_{i}" for s in (0.0025, 0.005) for i in range(6)]
order = [n for n in order if n in R]
S = ("none", "D3", "D5", "W3", "R3", "X3", "A16")
for k in (3, 4, 5):
    for V in ("V1", "V2"):
        print(f"\n### k={k} {V}: bias pp (s.e.)\n| model | acc | " + " | ".join(S) + " |\n|---|---|" + "---|" * len(S))
        for n in order:
            b, se = R[n]["bias_pp"], R[n]["se_pp"]
            print(f"| {n} | {R[n]['acc_true']:.4f} | " + " | ".join(f"{b[f'{s}_k{k}_{V}']:+.3f} ({se[f'{s}_k{k}_{V}']:.3f})" for s in S) + " |")
        print("| mean abs | | " + " | ".join(f"{np.mean([abs(R[n]['bias_pp'][f'{s}_k{k}_{V}']) for n in order]):.3f}" for s in S) + " |")
        print("| max abs | | " + " | ".join(f"{np.max([abs(R[n]['bias_pp'][f'{s}_k{k}_{V}']) for n in order]):.3f}" for s in S) + " |")
print("\nitineraries D3 (600k):", {n: R[n]["n_itin_D3"] for n in order})
print("sets:", {n: R[n]["sets"] for n in order})
p = "D3_k4_V1"; x = "X3_k4_V1"
b = np.array([R[n]["bias_pp"][p] for n in order])
gain = sum(abs(R[n]["bias_pp"][p]) < abs(R[n]["bias_pp"][x]) - 2 * np.hypot(R[n]["se_pp"][p], R[n]["se_pp"][x]) for n in order)
c1 = np.all(np.abs(b) <= 0.08); c2 = gain >= 12
print(f"\nPRIMARY D3 k4 V1 over {len(order)} models: max|bias| {np.abs(b).max():.3f}, n within .08: {(np.abs(b) <= .08).sum()}; beats X3 on {gain}")
part = [f"{s}_k{k}_{V}" for s in ("D3", "D5", "W3") for k in (3, 4, 5) for V in ("V1", "V2")
        if f"{s}_k{k}_{V}" != p and all(abs(R[n]["bias_pp"][f"{s}_k{k}_{V}"]) <= .08 for n in order)]
print("VERDICT:", "PASS" if c1 and c2 else ("PARTIAL " + str(part) if part else "FAIL"))
