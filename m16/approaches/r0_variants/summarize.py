"""Summary over the DEV2 models in results/biases.json: RMS, noise-corrected RMS, mean, slope on truth, region split."""
import json
import numpy as np

import sys
SRC = sys.argv[1] if len(sys.argv) > 1 else "results/biases.json"
B = json.load(open(SRC))
T = json.load(open("/home/user/dl/dev2/truths.json"))["truths"]
names = list(B)
acc = np.array([100 * T[n]["acc"] for n in names])
out = {"n_models": len(names)}
print(f"{len(names)} models; truth acc mean {acc.mean():.2f} pp")
print("| variant | dim | RMS pp | RMS noise-corr | mean pp | slope pp/pp | max abs | blind | 2~3 | 1~2 | tail (mean / RMS) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
DIM = {"a_R0": 3, "b_top4": 4, "c_median": 3, "dprime_spread": 3, "e_R0_latest": 4, "x_top4_latest": 5, "x_top4_spread": 4}
for v in B[names[0]]:
    if v == "acc_sample":
        continue
    b = np.array([B[n][v]["bias_pp"] for n in names])
    se = np.array([B[n][v]["se_pp"] for n in names])
    rms = float(np.sqrt(np.mean(b ** 2)))
    rmsc = float(np.sqrt(max(np.mean(b ** 2) - np.mean(se ** 2), 0)))
    slope = float(np.polyfit(acc, b, 1)[0])
    parts = {r: np.array([B[n][v][f"{r}_pp"] for n in names]) for r in ("blind", "band23", "band12", "tail")}
    out[v] = dict(dim=DIM[v], rms=rms, rms_corr=rmsc, mean=float(b.mean()), slope=slope, max_abs=float(np.abs(b).max()),
                  mean_se=float(se.mean()), **{f"{r}_mean": float(p.mean()) for r, p in parts.items()},
                  **{f"{r}_rms": float(np.sqrt(np.mean(p ** 2))) for r, p in parts.items()})
    o = out[v]
    print(f"| {v} | {o['dim']} | {rms:.3f} | {rmsc:.3f} | {o['mean']:+.3f} | {slope:+.4f} | {o['max_abs']:.2f} | "
          + " | ".join(f"{o[r+'_mean']:+.2f} / {o[r+'_rms']:.2f}" for r in ("blind", "band23", "band12", "tail")) + " |")
json.dump(out, open(SRC.replace("biases", "summary"), "w"), indent=1)
