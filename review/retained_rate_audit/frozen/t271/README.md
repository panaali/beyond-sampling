# rate_alpha — T271: convergence rates of the deterministic families, and an active-fraction-matched random control

↑ [TASKS](../../../TASKS.md) · [Fable review](../../../review/fable_review/README.md) (A4, A5, C2, D6) · [SCALING_OBSTRUCTIONS](../../SCALING_OBSTRUCTIONS.md)

**All claims unverified.** Diagnostic only.

## Pre-registration (written 2026-10-18, before any T271 number was computed)
**Question 1 (A4/C2).** For each deterministic family, how does its error shrink with compute? We fit alpha in
|bias| ~ C^-alpha (and, for families with a rule seed, RMSE over seeds ~ C^-alpha_rmse). Sampling has alpha_rmse = 0.5.
**Rule (DRI):** alpha <= 0.5 closes a family; alpha > 0.5 earns prefactor work. alpha is the OLS slope of log|bias| on log C
over >= 4 budgets spanning >= 16x. The CI is the slope ± t_{n-2, .975} x SE. A family's verdict uses the target's alpha and
the pooled median over models.

**Models:** target M_16,10 (truth = T192's 3e11-sample answer key 0.9531174789 ± 3.9e-7) and 4 leaderboard noisy copies
s0.005_0, s0.005_1, s0.01_0, s0.01_1 (truth = leaderboard 3.2e7-sample truths, SE 4-7e-5 = the bias resolution there).

**Families and their budget knob** (the knob must stay *within* one method; the v2 -> v3 -> v4 sequence in FINDINGS §5 is
three different methods, so its bias drop .74 -> .59 -> .25 pp is not a convergence rate):
- **F1 sci_tao** (fixed scrambled-Sobol face rule; `m16/approaches/sci_tao/estimator.py`, unchanged): budgets 4k, 16k,
  64k, 256k, 1M FPE (250x); 8 independent rule scrambles per budget. Bias = mean error over scrambles; RMSE over scrambles.
  Cost = charged FPE (x2.23 audit factor).
- **F2 probe v2 and v4** (`m16/estimate.py`, unchanged; weights swapped by patching its `run`/`Wo`): knob = the number of
  canonical arrangements averaged per configuration, s = 1, 2, 4, 8, 16 (cost in model evals x s). Estimate = the mean of the
  per-arrangement estimates (v4: the mean over seeds 0..s-1; v2: its own `seeds=` argument).
- **F3 FORM/SORM** (`m16/approaches/reliability/estimator.py`, unchanged): knob = `n_arr` (chamber arrangements = design
  points) 1, 2, 4, 8, 16, cost as the estimator reports.
- **F4 mass-truncated exact enumeration**, T = 4 (zoo (8,4) and (16,4) seed 0; T206 per-leaf exact data): keep the N heaviest
  leaves, count unkept mass as correct (bias = error mass of the unkept leaves). Cost = the kept leaves' mass ops + the
  mean geometry ops per leaf x N. This is an idealised best-first order (ordering by exact mass is free here). N spans >= 64x.
  (No T = 5 per-leaf data exists; noted.)
If a family is too slow for all budgets within the ~3 CPU-h cap, run >= 4 budgets spanning >= 16x on the target first and
report what was skipped.

**Question 2 (A5/D6).** Is the trained net's region count generic once the random controls match its active fraction?
Positive rescaling does not change gate patterns in a bias-free ReLU net, so activity cannot be matched by scale. Instead:
draw a pool of 400 Frobenius-matched i.i.d. random models (seeds 1000+), measure the T = 10 active fraction (1e5 inputs, the
T232 definition: the mean of h_T > 0) and keep the 5 draws closest to the trained 0.773 (within ±0.03). If fewer than 5 qualify,
use W_hh entries N(mu, 1) with mu chosen on a grid so the pool median activity is 0.77, Frobenius-rescaled. Count exact
prefix regions k = 2, 3, 4.
**Rule:** if matched-random k = 4 regions >= 1.3x trained, then training *reduces* the count relative to equally active random
nets (T232's "generic" verdict is revised). Within ±30% the T232 verdict stands. Below 0.77x, trained has more.

## Method
`rates.py fam` (sci_tao | probe | form | enum) appends one row per (model, budget, scramble) to rows.jsonl. Every estimator is
imported unchanged; the probe estimator's weights are swapped by patching `estimate.run`/`Wo`. `fit.py` -> fit.json / fit.out does
the OLS fits, measures the MSE ratio vs sampling at each budget (MSE = RMSE^2 over scrambles, or bias^2 for single-rule
families; cost converted to samples: FPE / model evals = 1 sample each; estT ops via T196 2 x ops / c_s(H,4)) and predicts the
budget range within 10x from the fitted law (TASKS directive 8: both bars reported). `d6.py` does Question 2. A container restart
interrupted the first run; the runner resumes from rows.jsonl, and no row was recomputed or edited.

## Results (unverified)
### Question 1: rates
| family | model | budget span | alpha_bias [95% CI] | alpha_rmse [95% CI] | measured MSE ratio (range) | budgets <= 1 | budgets <= 10 | predicted within 10x |
|---|---|---|---|---|---|---|---|---|
| enum_T4 | zoo16_4 | x95 | 1.21 [0.67, 1.76] | — | 0.0124-27 | 1/7 | 3/7 | C >= 9.74e+07 |
| enum_T4 | zoo8_4 | x63 | 1.25 [0.59, 1.90] | — | 0.38-928 | 1/7 | 1/7 | C >= 4.36e+08 |
| form | orig | x16 | 0.21 [-0.53, 0.94] | — | 45.4-580 | 0/5 | 0/5 | C <= 6.52e+04 |
| form | s0.005_0 | x16 | 0.12 [-0.13, 0.36] | — | 1.85e+03-1.66e+04 | 0/5 | 0/5 | C <= 3.68e+03 |
| form | s0.005_1 | x16 | -0.01 [-0.25, 0.23] | — | 4.11e+03-5.23e+04 | 0/5 | 0/5 | C <= 9.09e+03 |
| form | s0.01_0 | x16 | -0.02 [-0.11, 0.07] | — | 4.5e+03-7.06e+04 | 0/5 | 0/5 | C <= 7.49e+03 |
| form | s0.01_1 | x16 | -0.17 [-0.36, 0.02] | — | 2.11e+03-6.86e+04 | 0/5 | 0/5 | C <= 4.64e+04 |
| probe_v2 | orig | x16 | 0.58 [-0.12, 1.29] | — | 5.15-55.7 | 0/5 | 1/5 | C >= 1.4e+06 |
| probe_v2 | s0.005_0 | x16 | 0.57 [-0.99, 2.12] | — | 0.215-21.7 | 2/5 | 4/5 | C >= 5.67 |
| probe_v2 | s0.005_1 | x16 | -0.07 [-0.24, 0.10] | — | 378-1.05e+04 | 0/5 | 0/5 | C <= 344 |
| probe_v2 | s0.01_0 | x16 | 0.01 [-0.31, 0.33] | — | 23.2-287 | 0/5 | 0/5 | C <= 5.61e+03 |
| probe_v2 | s0.01_1 | x16 | -0.95 [-2.47, 0.58] | — | 0.0243-235 | 1/5 | 1/5 | C <= 3.11e+04 |
| probe_v4 | orig | x16 | -0.13 [-0.74, 0.47] | — | 11.6-374 | 0/5 | 0/5 | C <= 1.39e+05 |
| probe_v4 | s0.005_0 | x16 | -0.11 [-1.06, 0.83] | — | 9.09-586 | 0/5 | 1/5 | C <= 6.46e+04 |
| probe_v4 | s0.005_1 | x16 | 0.02 [-0.19, 0.22] | — | 8.82e+03-1.52e+05 | 0/5 | 0/5 | C <= 125 |
| probe_v4 | s0.01_0 | x16 | 0.11 [-0.36, 0.57] | — | 388-3.3e+03 | 0/5 | 0/5 | C <= 1.82e+03 |
| probe_v4 | s0.01_1 | x16 | -0.17 [-0.75, 0.41] | — | 71.3-4.42e+03 | 0/5 | 0/5 | C <= 2.26e+04 |
| sci_tao | orig | x246 | 0.49 [-0.37, 1.35] | 0.47 [0.32, 0.62] | 0.433-1.05 | 4/5 | 5/5 | C <= 1.26e+24 |
| sci_tao | s0.005_0 | x245 | 0.55 [0.17, 0.94] | 0.63 [0.50, 0.77] | 0.429-1.98 | 2/5 | 5/5 | C >= 30.3 |
| sci_tao | s0.005_1 | x248 | 0.62 [0.38, 0.86] | 0.56 [0.51, 0.61] | 1.38-2.53 | 0/5 | 5/5 | C >= 0.112 |
| sci_tao | s0.01_0 | x246 | 0.47 [-0.15, 1.10] | 0.73 [0.53, 0.93] | 0.285-5.21 | 2/5 | 5/5 | C >= 2.06e+03 |
| sci_tao | s0.01_1 | x246 | 0.58 [0.08, 1.08] | 0.57 [0.31, 0.84] | 1.2-5.76 | 0/5 | 5/5 | C >= 2.49 |

Ranges are in each family's cost unit (FPE or model evals; enum: estT ops). For alpha below 0.5 the ratio grows with C, so the
range is an upper bound on C. Noisy-copy biases below 2x the truth SE (~1e-4) are unresolved; fit.json repeats each fit
without them.

**Reading.**
1. **Probe v2 and v4: no rate within a method** (alpha_bias -0.95..0.58, all CIs include 0; v4 on the target -0.13). Averaging
   1 -> 16 canonical arrangements leaves the bias where it is (target v4 +0.26 .. -0.35 pp). Copy s0.005_1 is off by -7.6 to -11 pp
   at every budget. The FINDINGS §5 sequence .74 -> .59 -> .25 pp is a comparison of three *methods*, not a convergence rate.
   **Closed** (alpha <= 0.5). Measured ratios are 0.02-1.5e5 and erratic. Within 10x on 7 of 50 (model, budget) points, scattered, with no trend in budget (v2 on s0.005_0 4/5, v2 s0.01_1 1/5, v4 s0.005_0 1/5, v2 orig 1/5); <= 1 on 3 of 50.
2. **FORM/SORM: alpha ~0** (-0.17..0.21). The bias is a fixed offset of +0.5 .. +2.4 pp on the copies and is not reduced by 16x more
   design points. Ratios are 45-7e4. **Closed.**
3. **sci_tao: alpha_rmse 0.47-0.73**, i.e. the RQMC rate of ~0.5 (CIs include 0.5 on every model). Its MSE ratio therefore stays
   roughly constant with budget: **0.43-1.05 on the target** (<= 1 at 4 of 5 budgets, over 8 scrambles) and **0.29-5.8 on the
   copies** (within 10x at every budget, <= 1 at some). The lenient bar is met everywhere measured, the strict one only on the
   target and only by a 1-2x margin, which does not grow with budget. Not closed by the rule (alpha_rmse > 0.5 on 4 of 5 models),
   but alpha ~0.5 means no budget at which it pulls ahead. Its prefactor is the whole story (T223).
4. **Mass-truncated exact enumeration (T = 4): alpha ~1.2** [CI 0.6-1.9], the only super-0.5 rate. It is a finite method,
   though: bias reaches 0 at the full leaf set, and the fit describes the approach to exactness. Measured ratio <= 10 from 1/8 (16,4) or 1/2 (8,4) of the leaves,
   and <= 1 at half of them (0.012 at (16,4), 0.38 at (8,4)). This is exact enumeration in mass order,
   and its total cost scales with the region count (§3.2 of SCALING_OBSTRUCTIONS).
5. **Per LB3 group, both bars** (target = orig; s0.005 and s0.01 = two copies each): only sci_tao is within 10x in all three groups;
   it is <= 1 in orig only. Probe and FORM are outside 10x at almost every point in every group (exceptions above), with no budget trend.

### Question 2: active-fraction-matched random control (D6)
The i.i.d. Frobenius-matched pool (400 draws) maxes out at activity 0.749 (median 0.50) vs the trained 0.773, so only one draw
is within ±0.03. Per the pre-registered fallback, W_hh entries are N(0.25, 1), Frobenius-rescaled (grid: mu = 0 / .1 / .2 / .3
gave median activity .51 / .61 / .69 / .82), and the 5 draws of 100 closest to 0.773 are kept.

| draw | activity | regions k = 2 / 3 / 4 |
|---|---|---|
| mu3037 | 0.772 | 34 / 720 / 15,958 |
| mu3047 | 0.765 | 34 / 722 / 16,378 |
| mu3042 | 0.783 | 34 / 768 / 19,314 |
| mu3060 | 0.756 | 34 / 698 / 15,012 |
| mu3064 | 0.756 | 34 / 760 / 18,634 |
| 1396 | 0.749 | 34 / 810 / 21,702 |
| trained target | 0.773 | 34 / 792 / 21,032 |

Matched draws: k = 4 median 16,378, i.e. **0.78x the trained count** (range 15,012-19,314). Rule: within ±30% -> **T232's verdict
stands** (borderline, at 0.78x: activity-matched random nets have, if anything, slightly *fewer* regions). The best i.i.d.
draw (activity 0.749) has 21,702. So matching activity does not reveal a training-induced reduction. The DRI_HANDBOOK line
"trained nets use a tiny fraction of possible regions" is refuted either way [measured]. Caveat: the mu-shift changes the
ensemble (positive mean coupling), not only the activity.

## Claims (unverified)
- [measured] Within-method rates: probe v2/v4 alpha ~0, FORM/SORM alpha ~0, sci_tao alpha_rmse ~0.5, mass-truncated
  enumeration ~1.2 (finite method).
- [inferred] No deterministic family measured here has a rate that would overtake sampling at large budgets. sci_tao matches
  sampling's rate with a prefactor of ~0.4-1 on the target and 1-6 on copies.
- [measured] Activity-matched random weights give 0.71-0.92x the trained k = 4 region count.

## Compute accounting
sci_tao 200 runs (~25 CPU-min), probe 50 v2/v4 points (~25 CPU-min), FORM 25 runs (~30 CPU-min), enum from saved data, D6 pool
+ counts (~10 CPU-min). ~1.5 CPU-h total, single thread.
