"""T282 summary: per-group RMS bias (pp) per design and m, b_max(256k) per group, the pre-registered verdicts. -> summary.json"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
F = json.load(open(os.path.join(HERE, "frozen.json"))); TR = json.load(open(os.path.join(ROOT, "results", "leaderboard_truths.json")))
GR = ["orig", "seed", "s0.0025", "s0.005", "s0.01", "s0.02"]
grp = lambda n: "orig" if n == "orig" else "seed" if n.startswith("seed") else n.rsplit("_", 1)[0]
THEIRS_FULL0 = dict(zip(GR, (0.96, 1.84, 0.52, 1.31, 2.53, 3.45)))
bmax = {g: 100 * np.sqrt(10 * np.mean([TR[n]["acc"] * (1 - TR[n]["acc"]) for n in F if grp(n) == g]) / 256000) for g in GR}
keys = sorted({k for v in F.values() for k in v["bias"]})
rms = {k: {g: 100 * float(np.sqrt(np.mean([F[n]["bias"][k] ** 2 for n in F if grp(n) == g]))) for g in GR} for k in keys}
design = {g: 100 * float(np.sqrt(np.mean([F[n]["design_bias"] ** 2 for n in F if grp(n) == g]))) for g in GR}
print("b_max(256k)", {g: round(v, 2) for g, v in bmax.items()}); print("design_bias", {g: round(v, 2) for g, v in design.items()})
for d in ("full", "triple", "V1_sign", "V2_mix"):
    for m in range(9):
        print(f"{d:8s} m={m}", " ".join(f"{rms[f'{d}_m{m}'][g]:5.2f}" for g in GR))
agree_a = sum(abs(rms["full_m0"][g] / THEIRS_FULL0[g] - 1) <= 0.25 for g in GR)
mins = {d: {g: min(rms[f"{d}_m{m}"][g] for m in range(5)) for g in GR} for d in ("full", "triple", "V1_sign", "V2_mix")}
dead_b = all(mins[d][g] > 3 * bmax[g] for d in ("full", "triple") for g in ("orig", "seed"))
var = {d: dict(go=all(mins[d][g] <= bmax[g] for g in GR), dead=any(mins[d][g] > 3 * bmax[g] for g in GR)) for d in ("V1_sign", "V2_mix")}
print(f"(a) full-order m=0 within 25% of T274b on {agree_a}/6 groups; (b) dead line (orig & seed, full+triple): {dead_b}")
print("variants:", var, {d: {g: round(v, 2) for g, v in mins[d].items()} for d in ("V1_sign", "V2_mix")})
json.dump(dict(bmax=bmax, rms=rms, design_bias=design, min_m4=mins, agree_a=int(agree_a), dead_b=bool(dead_b), variants=var,
               theirs_full0=THEIRS_FULL0), open(os.path.join(HERE, "summary.json"), "w"), indent=1)
