# frozen_cheap — T287: can witness/frozen-gate class estimators be made cheap enough for F <= 10 (Directive 8 amendment 2)?

↑ [TASKS](../../../TASKS.md) · [frozen_check (T282)](../frozen_check/README.md) · T274b (lenient_bb)

**All claims unverified.**

## Pre-registration (written 2026-10-18, before any T287 number was computed)
**Score (DRI, Directive 8 amendment 2).** F = mean(e^2) x max(C, 1k FPE) / mean(p(1-p)) per LB3 group. Here e is the estimator's
error on each model, C its charged cost in FPE (1 FPE = 6000 FLOPs at (16,10); instrumented, every weight-dependent op), and
p the model's accuracy (leaderboard 3.2e7-sample truths). Strict F <= 1, lenient F <= 10.

**Estimator E1(design)** (deterministic): estimate = sum over classes c of w_c v_c.
- w_c: the class's exact model-free probability (combinatorics of i.i.d. N(0,1) orderings and sign counts; free model-free
  precomputation).
- x_c: a constructed witness. Model-free conditional means of the order statistics given the class's constraints, placed at the
  class's positions.
- v_c = 1 if the trained net (one forward pass of x_c) classifies x_c correctly, else 0.
- Cost: one forward pass per class (charged 1 FPE each), plus a negligible label/argmax.
- Labels (Hilton): w_c deductive; x_c constructed (allowed); v_c applied to the whole class is **inductive** (it extrapolates
  one input's verdict to the class). The frozen-gate "perfect integration" numbers below are the deductive ideal of the same
  class design (diagnostic, measured by paired sampling as in T282).

**Designs (refining classes):**
- D0 = (top-2 positions, k = number of positives): 90 x 11 = 990 classes.
- D1 = (top-3 positions, k): 720 x 11 = 7,920 (T274b's rank_exact classes, my arrangement).
- D2 = D1 x (signs of x_8, x_9) where consistent: <= 31,680.
- D3 = D1 x (bin of x_(2) - x_(3) in 2 model-free halves at its conditional median given k): 15,840.
- Witness variant W (the DRI's "weight-derived"): for D1, the gate pattern at the class's conditional-mean input propagated
  with frozen gates. This equals the witness verdict at m = 0, so it is not a separate estimator. Reported as the frozen-gate
  ideal (perfect integration) per design.
- Exact first steps (diagnostic, D1 classes, perfect integration): steps 1..f exact, f+1..10-m frozen at the witness gates, last m
  exact, for f = 1..4 and m = 0, 2, 4.

**Go line.** A frozen E1 design passes iff F <= 10 on all six LB3 groups at its C_eff = max(C, 1k). If any passes, it is frozen
as est(model, budget) -> (estimate, cost_flops) for HP2 scoring. If none passes, the (cost, RMS) curve and the per-group F are
the result, together with the perfect-integration F_ideal = RMS_ideal^2 x C_eff / p(1-p), which says whether better integration
could rescue the design. Repeat on DEV2 when it lands.

## Method
`cheap.py` -> cheap.json (E1 estimates for all 30 LB3 models; frozen-gate ideal for D1 by paired sampling, 400k inputs per
model, T282 protocol: the input is moved into the witness arrangement and scored against the net on the moved input).
`summary.py` -> summary.json (F per group at C_eff = max(C, 1k FPE), truths = leaderboard 3.2e7-sample). Class weights are
exact combinatorics (each design's weights sum to 1 within 7e-14). Witness values are model-free conditional means of the order
statistics (4e6 draws). The D2 class count is 15,312 after consistency, below the pre-registered upper bound of 31,680.

## Results (unverified)
### E1 (witness verdict per class; RMS pp / F per LB3 group)
| design | cost (FPE) | orig | seed | s0.0025 | s0.005 | s0.01 | s0.02 | go (F <= 10 all) |
|---|---|---|---|---|---|---|---|---|
| E1_D0 | 990 | 3.76 / 31.7 | 6.45 / 52.2 | 5.52 / 51.2 | 8.21 / 74.4 | 8.23 / 43.8 | 11.96 / 64.6 | no |
| E1_D1 | 7,920 | 3.59 / 228 | 6.42 / 410 | 5.19 / 357 | 8.12 / 577 | 10.25 / 539 | 11.95 / 511 | no |
| E1_D2 | 15,312 | 3.63 / 452 | 6.46 / 804 | 5.28 / 715 | 8.09 / 1.11e+03 | 10.74 / 1.14e+03 | 12.57 / 1.09e+03 | no |
| E1_D3 | 15,840 | 3.30 / 386 | 5.71 / 649 | 4.18 / 464 | 5.32 / 495 | 5.85 / 351 | 7.19 / 370 | no |

Mean bias (pp; positive = overestimate):

| design | orig | seed | s0.0025 | s0.005 | s0.01 | s0.02 |
|---|---|---|---|---|---|---|
| E1_D0 | +3.76 | +6.18 | +5.44 | +7.95 | +7.28 | +7.51 |
| E1_D1 | +3.59 | +6.22 | +5.12 | +7.87 | +10.16 | +10.21 |
| E1_D3 | +3.30 | +5.57 | +4.17 | +5.27 | +5.47 | +5.41 |

### Frozen-gate ideal (deductive integration of the same D1 classes, perfect; RMS pp / F if it cost 7,920 FPE)
| exact steps | orig | seed | s0.0025 | s0.005 | s0.01 | s0.02 |
|---|---|---|---|---|---|---|
| f=0, m=0 (10 frozen) | 0.69 / 8.34 | 2.30 / 52.6 | 0.43 / 2.49 | 1.57 / 21.7 | 2.73 / 38.3 | 3.50 / 43.7 |
| f=0, m=4 (6 frozen) | 1.06 / 19.8 | 1.81 / 32.6 | 0.92 / 11.2 | 0.82 / 5.9 | 0.61 / 1.93 | 0.51 / 0.942 |
| f=2, m=4 (4 frozen) | 1.05 / 19.6 | 1.71 / 29.1 | 0.91 / 11 | 0.81 / 5.75 | 0.61 / 1.9 | 0.52 / 0.975 |
| f=3, m=4 (3 frozen) | 0.84 / 12.6 | 1.09 / 11.9 | 0.70 / 6.5 | 0.67 / 3.92 | 0.46 / 1.1 | 0.41 / 0.598 |
| f=4, m=0 (6 frozen) | 0.44 / 3.49 | 1.31 / 17.2 | 0.39 / 2.03 | 1.52 / 20.2 | 2.67 / 36.5 | 3.48 / 43.4 |
| f=4, m=2 (4 frozen) | 1.27 / 28.6 | 1.24 / 15.3 | 1.04 / 14.3 | 0.74 / 4.76 | 0.63 / 2.07 | 1.01 / 3.67 |
| f=4, m=4 (2 frozen) | 0.52 / 4.87 | 0.61 / 3.67 | 0.44 / 2.56 | 0.45 / 1.79 | 0.26 / 0.347 | 0.30 / 0.32 |

## Verdict (pre-registered go line): **no setting passes; nothing to freeze**
- **E1 fails by 3-110x on F.** The witness-verdict step makes 3-12 pp RMS errors (always *over*estimates, +3.3 to +10.2 pp in every group: a class's
  canonical witness is a typical, well-separated input, so it is classified correctly even where the class has many errors).
  Refining the classes (D0 -> D1 -> D2) does not reduce the error, so F just grows with cost. Splitting by the 2-3 gap (D3) helps a little
  (RMS 3.3-7.2). The cheapest, D0 at 990 FPE, has F = 32-74.
- **Even perfect integration of the frozen-gate classes is not enough** unless almost every step is exact. With D1 classes, F <= 10
  on all six groups only with 4 exact first + 4 exact last steps (2 frozen): F = 4.9 / 3.7 / 2.6 / 1.8 / 0.35 / 0.32. Exact
  integration of an 8-step exact network is the region problem again (T222). With 3 frozen steps (f = 3, m = 4) orig and seed already fail (12.6 /
  11.9).
- Freezing step 1 costs nothing (f = 0 and f = 1 are identical: step-1 gates are set by sign x_0, which the witness matches).
- **Directive 8 amendment 2 changes the target from "1 pp at 256k" to "~2 pp at 1k" for the lenient bar,** but this family's
  cheap form (witness verdicts) is ~4-12 pp off, and its accurate form is not cheap.

## Alternatives not ruled out
- Witnesses chosen to minimise the class's verdict error (e.g. several witnesses per class with mass-weighted verdicts): V2 in
  T282 (quantile mixtures) did not help the frozen ideal; not tried for verdict-only E1.
- Classes defined by the network's own early gate patterns rather than input order statistics.
- Classes split by more than the 2-3 gap (1-2 gap, blind case): D3's small gain suggests gap bins matter most.

## Compute accounting
E1: 4 designs x 30 models x <= 15,840 witness forward passes (seconds). Ideal diagnostics: 30 models x 400k x 15 settings (~35 CPU-min).

### DEV2 (80 open retrains, ARC's scoring distribution; `dev2.py` -> dev2.json, dev2.log)
Weights: T192 branch 611796bd gpu_cost_law/hp2/dev2/weights.npz (read from git; only the manifest is copied here). The panel's open
truths.json had not landed, so the truths are **provisional 2e6-sample MC** per model (SE ~2e-4, subtracted). That is ample at
pp-scale errors. Rescore when the open truths land.

| design | cost (FPE) | RMS (pp, 80 models) | F |
|---|---|---|---|
| E1_D0 | 990 | 5.89 | 47.9 |
| E1_D1 | 7,920 | 6.24 | 425 |
| E1_D2 | 15,312 | 6.27 | 829 |
| E1_D3 | 15,840 | 5.45 | 648 |

Same picture as LB3: every witness-verdict design overestimates by 5-9 pp on the retrains, F >= 48. **No go on DEV2.**
