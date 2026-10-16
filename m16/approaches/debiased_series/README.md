# debiased_series: turning the biased probe series into an unbiased estimator (MLMC / Rhee-Glynn)

## Question
The deterministic probe estimators v2 -> v3 -> v4 (FINDINGS sec. 5) converge slowly. Their bias falls from
0.74 to 0.59 to 0.25 pp while the cost roughly doubles per order. Can randomized truncation (Rhee & Glynn 2015,
McLeish 2011) or multilevel MC (Giles 2015) turn this hierarchy into an **exactly unbiased** estimator? Would
that estimator beat plain sampling, and stratified.py, at 16k, 64k and 256k forward-pass equivalents?

## Method
**Two-level estimator.** `acc = E[g] + E[f - g]`. Here f(x) is 1 when the model is correct, and g is a
per-input surrogate built from the probes. `ds.py` and `surrogate.py` hold the code.
- **The surrogate g.** For each of the 720 (p1,p2,p3) triples, a vectorised bisection on a canonical input
  finds the relative-gap thresholds tau23 and tau12 at which the given model flips (the v2 probes; `iters`
  steps on [0, 0.6]). This costs 720 x 2 x iters forward passes: 4,320 at iters=3 and 7,200 at iters=5.
  Then g(x) = sig((g23 - tau23)/(0.6 tau23)) * sig((g12 - tau12)/(0.6 tau12)), using x's triple and
  relative gaps. g is a lookup costing 0.037 forward-equivalents (measured by timing).
- **Level 0, E[g], is exact and model-free given tau.** The triple is independent of the order-statistic
  values, and tau lies on the bisection grid. So E[g] = mean over triples of H[tau23_k, tau12_k], where H
  is a 2^iters x 2^iters table computed once from 16M model-free inputs (`build_table`). The check against
  brute force (`check_table.py`): table 0.93129 vs brute force 0.93128 +- 0.00006 on the original, and
  0.84382 vs 0.84383 +- 0.00005 on a noisy copy. In the blind region g is constant, and that constant
  cancels.
- **Level 1, E[f - g].** Stratified sampling in the 7 strata of stratified.py. The residual uses a Neyman
  allocation from a design-time pilot on the original model (`alloc_it*.json`). The coefficient is fixed at
  beta = 1, so no beta is estimated and the estimator is exactly unbiased for any weights, up to the
  tabulation error of H (about 1e-5).

**Why there is no randomized truncation or extra levels (analysis).** Rhee-Glynn and MLMC pay off when the
exact level is unavailable or much more expensive than the approximations. Here the exact level f costs one
forward pass per sample, and every probe-order surrogate g_K costs a fixed, per-model *deterministic*
amount C_K plus a near-free lookup per sample. For any truncation level N, `E[g_N] + mean(f - g_N)` is
already unbiased. The MSE at budget B is V_K/(B - C_K), and randomizing N gives a mixture of these, so the
optimum is a single depth K*(B). Intermediate telescoping levels g_k - g_{k-1} are lookups whose means come
from the table anyway, so they add nothing. One variant was ruled out on paper: estimating E[g] by cheap
MC instead of the table (Giles two-level with c0 = 0.037 and V0 of about 0.04). That gives a
cost x variance product of (sqrt(V0 c0) + sqrt(V1))^2, about 0.045, which is no better than plain
sampling. Exploiting the triple/value independence to tabulate E[g] is what makes level 0 free.

**Side test (diag1.py), antithetic coupling.** Swapping the values at p2 and p3 (or p1 and p2), or
symmetrizing over all 6 permutations of the top-3 values, is measure-preserving. Cost-adjusted, it gives
**no gain**: variance 0.031 / 0.037 / 0.045 vs 0.033 for stratified alone on the original. The model
resolves even 1% relative gaps (comparator gain about 78), so the near-tie choice is not position-locked.

## Status
Done. **Negative to marginal result.** The construction is exactly unbiased, but the probe surrogate
explains little of the within-stratum variance.

## Results
Gain = plain-sampling MSE at equal cost / method MSE. Plain MSE = a(1-a)/cost, which is exact.

**Low-noise variance accounting** (`python variance_gain.py`, per-stratum s.d. from 300k inputs per model,
same fixed allocations as the estimators; geometric mean over models):

| set | budget | stratified | DS depth 3 (C=4.3k) | DS depth 5 (C=7.2k) | DS5 / stratified, range over copies |
|---|---|---|---|---|---|
| original | 16k | 2.07 | 1.39 | 1.12 | 0.54 |
| original | 64k | 2.07 | 1.78 | 1.80 | 0.87 |
| original | 256k | 2.07 | 1.87 | 1.97 | 0.95 |
| sigma 0.0025 (6) | 16k | 1.80 | 1.25 | 1.03 | 0.55-0.61 |
| sigma 0.0025 | 64k | 1.80 | 1.60 | 1.67 | 0.89-0.99 |
| sigma 0.0025 | 256k | 1.80 | 1.69 | 1.83 | 0.98-1.08 |
| sigma 0.005 (6) | 16k | 1.42 | 1.08 | 0.89 | 0.59-0.74 |
| sigma 0.005 | 64k | 1.42 | 1.38 | 1.44 | 0.95-1.19 |
| sigma 0.005 | 256k | 1.42 | 1.45 | 1.58 | 1.04-1.31 |

Extrapolated to infinite budget, DS5 / stratified is about 0.98 on the original, 1.04 at sigma 0.0025 and
1.14 at sigma 0.005. With per-model ideal Neyman allocation (diag3.py), the surrogate's residual variance
reduction over stratified is 1.03x on the original and 1.13-1.47x on noisy copies (largest on the 82%-accurate
copy).

**Repeat-based check of the actual estimators** (`python evaluate.py 0 16`, `python evaluate.py 0.0025 3`,
`python evaluate.py 0.005 3`; truth = 0.95317945 from 20M samples, and 4M for the noisy copies, the same
seed-12345 stream as arc_eval.true_acc). These MSEs come from 16 or 18 repeats, so each one is
uncertain to about +-35%. Use the table above for ranking.

| set | budget | stratified | DS3 | DS5 |
|---|---|---|---|---|
| original (16 reps) | 16k | 2.77 (bias +0.005+-0.026pp) | 1.64 (bias +0.012+-0.034pp) | 1.74 (bias +0.022+-0.032pp) |
| original (16 reps) | 64k | 4.58 (bias +0.007+-0.010pp) | 2.80 (bias -0.010+-0.013pp) | 1.57 (bias -0.016+-0.017pp) |
| original (16 reps) | 256k | 1.29 (bias -0.013+-0.009pp) | 1.52 (bias -0.009+-0.008pp) | 3.35 (bias -0.001+-0.006pp) |
| sigma 0.0025 (6x3) | 16k | 2.37 (bias +0.009+-0.030pp) | 1.39 (bias -0.015+-0.040pp) | 1.03 (bias +0.013+-0.046pp) |
| sigma 0.0025 (6x3) | 64k | 3.47 (bias -0.015+-0.012pp) | 1.56 (bias -0.028+-0.017pp) | 2.25 (bias -0.016+-0.015pp) |
| sigma 0.0025 (6x3) | 256k | 1.07 (bias -0.020+-0.010pp) | 1.17 (bias -0.022+-0.009pp) | 2.12 (bias -0.012+-0.008pp) |
| sigma 0.005 (6x3) | 16k | 1.23 (bias +0.096+-0.047pp) | 1.11 (bias +0.082+-0.051pp) | 1.03 (bias +0.080+-0.053pp) |
| sigma 0.005 (6x3) | 64k | 2.56 (bias -0.021+-0.017pp) | 1.83 (bias -0.032+-0.020pp) | 2.12 (bias -0.029+-0.019pp) |
| sigma 0.005 (6x3) | 256k | 0.52 (bias -0.021+-0.019pp) | 0.82 (bias -0.035+-0.013pp) | 1.09 (bias -0.017+-0.013pp) |

The direction agrees with the variance table: DS5 is worst at 16k and best at 256k. The magnitudes swing
widely (for example, stratified at 256k on sigma 0.005 shows 0.52, against 1.42 predicted), which is the
expected scatter of an 18-sample MSE. Do not read the 256k DS5 gains of 3.35 and 2.12 as real; the
variance table says 1.97 and 1.83.

**Unbiasedness on the original** (16 repeats per budget; bias in pp +- s.e.): all 9 method/budget cells lie
within 1.6 s.e. of 0 (largest: stratified at 256k, -0.0135 +- 0.0088 pp). On the noisy copies, the 16k
cells at sigma 0.005 sit at about +0.08-0.10 pp, about 1.5-2 s.e., for stratified and DS alike. That points
to shared sampling and truth noise rather than to DS. One cell is at 2.7 s.e.: DS3 at 256k on sigma 0.005,
-0.035 +- 0.013 pp. Among about 27 cells, one at 2.7 s.e. is plausible by chance (p about 0.2), but it has
not been re-run with fresh seeds.

## Compute accounting
- Counted: every model forward pass (probes C_K = 720 x 2 x iters, plus sampled inputs), plus 0.037 per
  sample for evaluating g. The 0.037 is a timing ratio of g against forward on 200k inputs.
- Not counted: input generation and rejection sampling for strata, the same as stratified.py.
- Model-free and free, computed once and reused across models: stratified_design.json P (20M inputs), and
  table_it{3,5}.npy (16M inputs each).
- Design-time pilot on the ORIGINAL model: alloc_it{3,5}.json (300k evals). It is not counted, the same
  convention as stratified_design.json. This slightly favours both methods on the original.
- Leaderboard entry `estimator.py`: stratified below 128k, depth 5 at 128k and above. Set DEDUCTIVE = False.

## Claims
1. [measured] The two-level estimator is unbiased in practice. The tabulated E[g] matches brute force to
   1e-5, and the empirical bias on the original is within 2 s.e. at every budget.
2. [inferred] It is exactly unbiased for any weights by construction (fixed beta = 1, exact E[g]).
3. [measured] The probe surrogate removes at most 1.03x (original) to 1.47x (worst noisy copy) of the
   stratified variance. After the probe cost it loses to stratified at 16k and 64k on every model. At 256k it
   gains 0.95-1.31x, averaging about 1.0 at sigma 0.0025 and about 1.1 at sigma 0.005.
4. [inferred] Randomized truncation and extra MLMC levels cannot help with this hierarchy. The exact level
   costs one evaluation per sample and the approximations cost a fixed amount per model, so the optimum is a
   single deterministic depth (analysis above).
5. [measured] Antithetic swaps of the top-3 values do not reduce variance at equal cost.
6. [inferred] The bottleneck is per-input: whether a near-tie flips depends on the other values through
   gating (FINDINGS sec. 3-4). Thresholds per triple and gap cannot see that. A deeper probe order (v3/v4
   sensitivities, 80-167k evals) would have to cut the residual variance by more than about 2x to pay at 256k.

## Alternatives not ruled out
- **Main claim: "a probe hierarchy cannot give a useful control variate."**
  - (a) The surrogate form is too crude. Hard thresholds and one logistic width (w = 0.6) were the only
    forms tried; w in {0.3, 0.6, 1, 2} was scanned in diag2/diag3. *Not tested:* first-order sensitivity
    thresholds (v3) as a per-input surrogate. E[g] would then need a non-tabulated integral.
  - (b) The variance is in the wrong place. Residual allocation from the original model is mis-specified
    for noisy copies. *Partly tested:* with per-model ideal Neyman allocation (diag3), DS gains 1.13-1.47x
    over Neyman-stratified on noisy copies, which is more than with the fixed allocation. An adaptive
    allocation, informed by the probes, might recover part of that difference. Not implemented.
  - (c) The canonical probe point is atypical for some copies (as v4 found). Not separately tested.
- The repeat-based MSEs are too noisy (16-18 repeats) to separate methods that differ by less than 1.5x.

## Failure modes
- At small budgets the fixed probe cost dominates. At 16k, DS is about 0.55x of stratified.
- If the surrogate is anti-correlated with f on some copy, beta = 1 can make the variance worse than
  stratified. Unbiasedness still holds.
- The tabulation error of H is fixed across repeats (about 1e-5, or 0.001 pp), which is negligible at
  these budgets.
