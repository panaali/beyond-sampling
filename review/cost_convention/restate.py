"""T196: restate published MSE ratios under one cost convention (pure arithmetic on saved results; < 1 CPU-second).

Recommended convention (README): FLOPs on both sides. multiply-add = 2 FLOPs; every other scalar operation
(add, compare, max/ReLU, abs, sign test, sqrt, div, atan2/asin/exp) = 1 FLOP. Sampling, per input of M_H,T:
  forward  2*T*H*(H+1)   (ARC/leaderboard formula: T steps of Wh h + Wi x_t, step 1 included)
  ReLU     T*H
  readout  2*T*H
  argmax   T-1           (over the T logits)
  label    2T-3          (top-2 selection on the T inputs, linear scan; not a full sort)
  compare  1, accumulate 1
  RNG      0             (as ARC, S8/S9)
  => c_s(H,T) = flops(H,T) + T*H + 3T - 2, flops = leaderboard.flops = T(2H^2+2H) + 2HT.
Estimator side: the per-cell counters (m43.Ops, est75, estT) count a multiply-add as 1 op and every other operation
as 1 op, so their FLOPs lie in [ops, 2*ops]. Until the counters split the two, charge 2*ops (upper bound = least
favourable to the estimator); the kappa = 1 column is the most favourable bound.
ratio = err^2 * N / p(1-p) with N = estimator cost / sampling cost, so a restated ratio = published * kappa*c_pub/c_s.
usage: python review/cost_convention/restate.py   (from the repo root)"""
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "m16"), os.path.join(ROOT, "m16", "approaches", "understand_m43")]
from leaderboard import flops, fpe_flops  # noqa: E402
from zoo44 import ops_per_sample  # noqa: E402


def c_s(H, T):            # recommended sampling charge, FLOPs per input
    return flops(H, T) + T * H + 3 * T - 2


def c_s_ops(H, T):        # same items in ops (multiply-add = 1), for the alternative convention
    return ops_per_sample(H, T) + T * H + 3 * T - 2


print("sampling charge per input")
print("| (H,T) | ops_per_sample (E36-E43) | fpe_flops (E46/E48/LB3) | flops() | recommended c_s FLOPs | same items in ops |")
print("|---|---|---|---|---|---|")
for H, T in ((4, 3), (5, 3), (6, 3), (8, 3), (8, 4), (8, 5), (16, 10)):
    print(f"| ({H},{T}) | {ops_per_sample(H, T)} | {fpe_flops(H, T):.2f} | {flops(H, T)} | {c_s(H, T)} | {c_s_ops(H, T)} |")

rows = []
U = os.path.join(ROOT, "m16", "approaches", "understand_m43")
for H in (4, 5, 6, 8):                                   # E43: worst seed per rung
    z = json.load(open(os.path.join(U, f"evalH_{H}.json")))["zoo"]
    s, v = max(z.items(), key=lambda kv: kv[1]["ratio"])
    rows.append(("E43", f"({H},3) zoo{s} noisy", v["ratio"], ops_per_sample(H, 3), H, 3))
F = os.path.join(ROOT, "m16", "approaches", "estT_fix")
summ = json.load(open(os.path.join(F, "summary.json")))   # E46: all 9 models
for n, v in summ.items():
    if "ratio" in v:
        rows.append(("E46", f"(8,4) {n}", v["ratio"], fpe_flops(8, 4), 8, 4))
ex, rc = (json.load(open(os.path.join(F, "run5", f"zoo0_{m}.json"))) for m in ("exact", "recipe108"))
p = ex["acc"]; r48 = (rc["acc"] - p) ** 2 / (p * (1 - p) / (rc["ops"] / fpe_flops(8, 5)))   # E48 recomputed
rows.append(("E48", "(8,5) zoo0", r48, fpe_flops(8, 5), 8, 5))

print("\nrestated ratios (restated = kappa=2 upper bound; kappa=1 = most favourable to the estimator)")
print("| experiment | model | published ratio | restated ratio | factor | restated, kappa=1 | ops-convention ratio |")
print("|---|---|---|---|---|---|---|")
for e, m, r, cpub, H, T in rows:
    f2 = 2 * cpub / c_s(H, T); f1 = cpub / c_s(H, T); fo = cpub / c_s_ops(H, T)
    print(f"| {e} | {m} | {r:.3g} | {r * f2:.3g} | {f2:.3f} | {r * f1:.3g} | {r * fo:.3g} |")
print("\nE36/E37 at (4,3) (published with 72 ops/sample): factor kappa=2", round(2 * 72 / c_s(4, 3), 3),
      "| kappa=1", round(72 / c_s(4, 3), 3), "| ops convention", round(72 / c_s_ops(4, 3), 3))
print("LB3 at (16,10): FPE 6000 FLOPs vs c_s", c_s(16, 10), "-> factor for estimators that divide FLOPs by 6000:",
      round(6000 / c_s(16, 10), 4), "| by 5760 (cone_enum):", round(5760 / c_s(16, 10), 4))
print("step-1 recurrence on h0=0 (2H^2 FLOPs) as a share of c_s:",
      {f"({H},{T})": round(2 * H * H / c_s(H, T), 3) for H, T in ((4, 3), (8, 3), (8, 4), (8, 5), (16, 10))})

# sensitivity: an efficient sampler skips the step-1 recurrence on h0 = 0 (2H^2 FLOPs); worst model per experiment
print("\nworst case (kappa = 2 AND sampling charged without the step-1 recurrence):")
for e in ("E43", "E46", "E48"):
    for _, m, r, cpub, H, T in [q for q in rows if q[0] == e]:
        cs_eff = c_s(H, T) - 2 * H * H
        print(f"  {e} {m}: {r * 2 * cpub / cs_eff:.3g}")
