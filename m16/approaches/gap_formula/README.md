# T269: gap-formula estimator (P(error | V) integrated over the exact order-statistic law)

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [T241 neuron account](../neuron_account/README.md)

## Question
Errors of M_16,10 live near ties and in the blind region. If P(err | V) depends on a few summaries V of the input,
accuracy = 1 − ∫ P(err | V) dLaw(V), and Law(V) is known exactly for 10 i.i.d. N(0,1) (positions uniform and
independent of the values; top-k values with closed-form density). Can a compact formula for P(err | V),
integrated by deterministic quadrature, estimate accuracy within 5e-4 with <= ~100 parameters?

## Pre-registration (written 2026-10-18 before any fit or scoring run)
**Summaries.** p1, p2, p3 = positions of the top 3; c = x9 case (max / 2nd / below); M1 > M2 > M3 = top-3 values;
g12 = M1 − M2, g23 = M2 − M3; blind = {M2 <= 0} (exact mass 11/1024).
- V1 = (g12, g23, c, ages of the top 3)
- V2 = V1 + (M1, M2, M3)
- V3 = V2 + K (number of positive inputs; its joint law with M3 is closed-form: given M3 = m3 > 0, K − 3 ~ Bin(7, (Φ(m3) − ½)/Φ(m3)))

**Formula.** Logistic P(err | V) on smooth functions of log-gaps (and, for V2/V3, log value ratios), with
coefficients per x9 case and, where needed, coarse age classes. The blind region is handled separately: exact mass
11/1024 times a measured blind error rate (1–2 parameters).
**Integration.** Tensor Gauss–Legendre in (M2, log g23, log g12) on M2 > 0, weighted by the exact density
720 φ(M1) φ(M2) φ(M3) Φ(M3)^7. Positions are summed exactly over their uniform law. Quadrature accuracy is checked by
convergence (node doubling), total mass = 1 − 11/1024, and recovery of known moments.
**Fitting data.** Fresh samples of the scored model's own inputs, split fit / held-out.

**Scoring.**
- Target M_16,10 against the answer key 0.9531174789 (3e11 samples, T192).
- Transfer to 5 zoo (16,10) seeds (`2nd_argmax_16_10_2^30_{0..4}`) and 2 noisy copies of the target (arc_eval
  noise, rel. sigma .005 and .01, seed 0, first copy). Their truths come from 1e8 fresh samples each (s.e. ~2e-5).
- Transfer modes:
  - (i) K = 0: the formula fit on the target, unchanged;
  - (ii) leave-one-out across the zoo seeds;
  - (iii) per-model refit of <= K = 3 parameters (per-case intercepts), on samples of the scored model.
- Every number states which mode it uses.

**Go criterion (from the DRI):** |bias| < 5e-4 with <= ~100 parameters. **Interpretation fixed in advance:** a
logistic fit with per-case intercepts reproduces its own training error rate, so on the model it was fit on, the
bias is just the fitting sample's s.e. (~1.5e-4 at 2M samples). That would pass trivially and show nothing about
the formula. The informative tests are therefore:
- (a) the K = 0 / leave-one-out transfers, which have no samples of the scored model;
- (b) whether the per-model refit beats plain sampling at equal sample count. That test is the regression/control-
  variate estimator: E[err] = ∫ f dLaw + mean(err − f), with variance about (1 − R²) of plain sampling.

Phase 1 "passes" only if (a) gives |bias| < 5e-4 on most transfer models, or (b) reduces variance by >= 4x
(so 1e9 FLOPs of samples give s.e. ≲ 2.5e-4). Phase 2 (coefficients from the weights) only if Phase 1 passes.

**Alternatives written first.**
1. The bias is dominated by the blind region (0.83% of the error mass), whose error rate depends on low inputs
   the formula does not see. Test: report the blind and non-blind contributions separately.
2. P(err | V) depends on low inputs (m4.., crosstalk; FINDINGS §3, H-E10). Integrating a V-conditional fit
   then commits a Jensen-type error only if the fit is mis-specified; exact integration of a calibrated fit is
   unbiased for its own model. The real risk is transfer: models differ in how they use low inputs. Test: V3 vs
   V2 gain, and transfer residuals.
3. Model-specific constants (comparator gain mismatch, T241 / FINDINGS §3) set the error rate, so no
   cross-model formula transfers at 5e-4. Test: K = 0 and leave-one-out errors vs the spread of the truths.
4. The quadrature, not the formula, limits the estimate (sharp near-tie features). Test: node doubling.

## Method
- `gfcommon.py`: the scored models (target, zoo seeds 0–4 exported from ARC's public `2nd_argmax_16_10_2^30_s.pth`
  into `zoo16_10_weights.npz`, two noisy copies built exactly like `arc_eval.noisy_models`), an exact float64
  forward pass, and the summaries.
- `truths.py`: truths from 2e7 fresh samples per model (s.e. 5–10e-5; the pre-registered 1e8 was cut for CPU). For
  the target we score against the answer key 0.9531174789; our 2e7 estimate is 0.953037 ± 4.7e-5.
- `phase1.py [families]`: per model, 1M fit + 1M held-out inputs (pre-registered 2M each, cut to fit in 2.5 GB).
  - Logistic fits with per-case blocks:
    - V1: [1, z23, z12, z23², z12², z23·z12] with z = log gap;
    - V2: scale-free [1, a, b, q, a², b², q², ab, aq, bq] with a = log(g23/M2), b = log(g12/M1), q = log(M2/M1);
    - V3: V2 + [K=2, K=3, min(K,8)/8].
  - Two blind error rates (K = 0, 1) times their exact masses 1/1024, 10/1024.
  - Integration: Gauss–Legendre in (M2 on (0, 6), log g23, log g12 on (−18, 2.5)), with 64/96/128 nodes per axis
    and K summed exactly for V3. The quadrature reproduces the non-blind mass 1 − 11/1024 to 6e-8, and E[M1·1{M2>0}]
    matches Monte Carlo (1.5311 vs 1.5307 ± 0.0007).
  - The CV estimate is ∫f dLaw − mean_heldout(err − f). Its variance ratio is var(err − f)/var(err) on held-out data.
- `report.py` -> `results/phase1_tables.md`.

## Status
**Done; Phase 1 fails its pre-registered criteria, so Phase 2 is not run.** Unverified (no independent check).

## Results
Reproduce: `python3 truths.py` (~17 CPU-min), `python3 phase1.py V1 V2` (~15 CPU-min), `python3 phase1.py V3`,
`python3 report.py`. Bias = estimate − truth.
- **own-fit**: formula fit on 1M samples of the same model;
- **CV var ratio**: control-variate variance relative to plain sampling at equal sample count;
- **K=0**: target formula applied unchanged;
- **K=3**: target formula with 3 per-case intercepts refit on 1M samples of the scored model;
- **LOO**: fit pooled on the other four zoo seeds.

**V1** (20 parameters incl. 2 blind rates)

| model | truth | own-fit bias | CV var ratio | K=0 transfer bias | K=3 refit bias | LOO bias |
|---|---|---|---|---|---|---|
| target | 0.95312 | -0.00013 | 0.714 | – | – | – |
| zoo0 | 0.90992 | +0.00006 | 0.808 | +0.04307 | -0.00005 | +0.00245 |
| zoo1 | 0.88025 | +0.00006 | 0.814 | +0.07275 | +0.00015 | +0.03963 |
| zoo2 | 0.94855 | +0.00002 | 0.721 | +0.00445 | +0.00003 | -0.04579 |
| zoo3 | 0.93209 | -0.00014 | 0.726 | +0.02090 | -0.00013 | -0.02515 |
| zoo4 | 0.88956 | +0.00012 | 0.805 | +0.06343 | +0.00020 | +0.02798 |
| noisy0.005 | 0.89185 | -0.00012 | 0.787 | +0.06114 | -0.00012 | – |
| noisy0.01 | 0.75782 | -0.00027 | 0.869 | +0.19517 | +0.21140 | – |

**V2** (32 parameters incl. 2 blind rates)

| model | truth | own-fit bias | CV var ratio | K=0 transfer bias | K=3 refit bias | LOO bias |
|---|---|---|---|---|---|---|
| target | 0.95312 | -0.00011 | 0.693 | – | – | – |
| zoo0 | 0.90992 | +0.00002 | 0.788 | +0.04308 | -0.00008 | +0.00250 |
| zoo1 | 0.88025 | +0.00007 | 0.804 | +0.07276 | +0.00015 | +0.03968 |
| zoo2 | 0.94855 | +0.00008 | 0.707 | +0.00446 | +0.00007 | -0.04575 |
| zoo3 | 0.93209 | -0.00018 | 0.695 | +0.02092 | -0.00018 | -0.02510 |
| zoo4 | 0.88956 | +0.00015 | 0.782 | +0.06344 | +0.00023 | +0.02805 |
| noisy0.005 | 0.89185 | -0.00014 | 0.766 | +0.06115 | -0.00019 | – |
| noisy0.01 | 0.75782 | -0.00024 | 0.857 | +0.19518 | -0.00040 | – |

**V3** (41 parameters; from `logs/phase1_V3.log`, whose JSON was lost when the LOO step ran out of memory):
own-fit bias −1.2e-4 (target) and |bias| ≤ 2.5e-4 elsewhere; CV variance ratio .694 (target), .695–.856 elsewhere;
K=0 and K=3 identical to V2 within 1e-5. So adding K buys nothing over V2.

Quadrature convergence (64 → 96 → 128 nodes per axis): estimates move by ≤ 1.2e-8 (V1) and ≤ 6.6e-6 (V2).

### Gap-conditioned error profile and the soft tail (DRI / fable review A2, D4; `python3 profile.py`, `python3 tailcheck.py`)
2M fresh inputs per model (seed 269300+i). Gap = min(g12, g23); "tail" = non-blind inputs with gap >= threshold.
The full 2-D table P(err | g23 bin, g12 bin) and the g23-only version are in `results/profile.json`.

| model | error % | P(err \| blind) | blind share of error | tail share of error (gap >= .05 / .1 / .2) | tail error, pp (>= .05 / .1 / .2) |
|---|---|---|---|---|---|
| target | 4.66 | 0.777 | 0.178 | 0.141 / 0.053 / 0.014 | 0.66 / 0.25 / 0.07 |
| zoo0 | 9.00 | 0.545 | 0.065 | 0.295 / 0.131 / 0.046 | 2.65 / 1.18 / 0.42 |
| zoo1 | 11.97 | 0.648 | 0.058 | 0.448 / 0.288 / 0.152 | 5.36 / 3.45 / 1.82 |
| zoo2 | 5.12 | 0.792 | 0.166 | 0.188 / 0.078 / 0.023 | 0.96 / 0.40 / 0.12 |
| zoo3 | 6.80 | 0.819 | 0.131 | 0.209 / 0.084 / 0.025 | 1.42 / 0.57 / 0.17 |
| zoo4 | 11.02 | 0.666 | 0.065 | 0.372 / 0.202 / 0.086 | 4.10 / 2.23 / 0.95 |
| noisy0.005 | 10.82 | 0.828 | 0.082 | 0.382 / 0.194 / 0.067 | 4.13 / 2.09 / 0.73 |
| noisy0.01 | 24.19 | 0.849 | 0.038 | 0.627 / 0.435 / 0.231 | 15.16 / 10.51 / 5.59 |

The target reproduces the review's estimate: 14.1% of the error (0.66 pp) sits at gap >= 0.05, non-blind. **The
weaker models differ from the target mostly in the tail.** It holds 19–45% of the error for the zoo seeds and
38–63% for the noisy copies, i.e. 1–15 pp. The near-tie and blind parts are much more alike across models. This is
the direct reason the K=0 / LOO transfers fail: the tail is model-specific.

**Does an own-model formula capture the tail?** V2 fit on 1M, scored on 1M held-out (seed 269400+i): error mass in
pp, observed / predicted, per gap bin:

| model | [0, .05) | [.05, .1) | [.1, .2) | >= .2 |
|---|---|---|---|---|
| target | 3.21 / 3.03 | 0.42 / 0.46 | 0.18 / 0.26 | 0.07 / 0.12 |
| zoo0 | 5.76 / 5.21 | 1.49 / 1.43 | 0.75 / 1.04 | 0.41 / 0.72 |
| zoo1 | 5.92 / 5.40 | 1.91 / 2.03 | 1.61 / 1.93 | 1.80 / 1.93 |
| zoo2 | 3.32 / 3.16 | 0.57 / 0.58 | 0.28 / 0.35 | 0.12 / 0.19 |
| zoo3 | 4.47 / 4.20 | 0.82 / 0.88 | 0.39 / 0.53 | 0.16 / 0.28 |
| zoo4 | 6.19 / 5.73 | 1.90 / 1.88 | 1.28 / 1.51 | 0.95 / 1.18 |
| noisy0.005 | 5.78 / 5.46 | 2.08 / 1.83 | 1.33 / 1.50 | 0.74 / 1.14 |
| noisy0.01 | 8.05 / 7.80 | 4.65 / 4.27 | 4.94 / 4.97 | 5.59 / 6.27 |

The logistic-in-log-gap shape is mis-specified. It under-predicts the near-tie bin by 0.18–0.55 pp and over-predicts
the far tail (>= .1) by 0.05–0.4 pp on every model. The total matches only because the per-case intercepts force
calibration. So even on its own model the formula does not *describe* the tail at the 5e-4 level; it trades errors
between bins. **Added to the go/no-go: fail** (the tail is not captured bin by bin, and it is where models differ).

**Verdict against the pre-registration:**
- Own-fit |bias| < 5e-4 on every model with 20–41 parameters. This is the trivial pass predicted in advance: the
  fitted formula reproduces its own 1M-sample error rate (s.e. ~2e-4). It does not count.
- **(a) Transfer fails by 1–3 orders of magnitude.**
  - K=0 errors are +0.4 to +19.5 pp. The target formula predicts the target's ~95% accuracy for every model.
  - LOO errors are 0.25–4.6 pp. They equal (mean accuracy of the other four seeds) − (this seed's accuracy), so the
    formula carries no model-specific information.
  - The truths themselves span 0.758–0.953.
- **(b) Variance reduction is small.** The CV variance ratio is .69–.87 (V1: .71–.87; V2/V3: .69–.86), i.e. a 1.15–1.45x
  gain, against the pre-registered >= 4x. At 1e9 FLOPs this moves sampling's s.e. from ~5e-4 to ~4.2e-4.
- The K=3 refits have |bias| ≤ 4e-4, except V1 on noisy0.01, where the offset-intercept refit diverged (+21 pp). But
  they need 1M samples of the scored model, i.e. they are sampling estimators with a fitted shape.

**Alternatives (pre-registered) — outcomes:**
1. Blind region dominates? **No.** Blind mass × error rate is at most 11/1024 ≈ 1.07 pp per model, far below the
   transfer errors (up to 19.5 pp).
2. Dependence on low inputs (V3 vs V2)? **No gain** from K (variance ratio .694 vs .693). The V-conditional variance
   that limits (b) is not removed by the positive-count summary. Other low-input summaries (m4, sum of the rest)
   were not tried.
3. Model-specific constants set the error rate? **Supported** [measured]: the same functional form needs different
   constants per model, at the pp level.
4. Quadrature limits? **No** (≤ 7e-6).

## Inductive vs mechanistic (DRI note, after ARC's framing)
Every Phase 1 estimate here is **inductive**: its coefficients are regressions fit on samples of a model (ARC: "a
little bit cheating"). The quadrature is deductive, but that only makes the estimator a deterministic function of
sampled coefficients. Only a Phase 2 formula whose coefficients are *derived from the weights* would count as
mechanistic, for example via the exact margin algebra in each gate pattern or T241's neuron formulas. Phase 2 was not
run (pre-registered stop), so **no mechanistic estimate exists from this task**.

## Compute accounting
Forward passes only (no LP): truths 1.6e8 sequences; Phase 1 runs 8 models × 2M sequences per run, about 4 runs
including two that ran out of memory. Logistic IRLS on ≤ 1M × 39 designs, quadrature on ≤ 128³ = 2.1M nodes per
integral. Total about 1.5 CPU-hours (incl. profile/tail check), single thread, ≤ 2 processes. The estimator itself (given the formula) is
deductive: one 3-D quadrature, no random inputs. But in Phase 1 its coefficients come from samples of the scored
model (sampling-based).

## Claims
- G1 [measured] Exact integration of a calibrated logistic P(err | V) over the order-statistic law reproduces each
  model's accuracy to sampling precision (|bias| ≤ 2.7e-4 with 1M fit samples). Quadrature error ≤ 7e-6.
- G2 [measured] A formula fit on one model does not transfer: errors are 0.4–19.5 pp to zoo seeds and noisy copies,
  and 0.25–4.6 pp leave-one-out across zoo seeds. Gaps, values, positions and K are not enough to carry the
  model-specific error rate.
- G3 [measured] As a control variate, P(err | V2) reduces variance by only 1.15–1.45x. Most of the error variance is
  not explained by (gaps, top-3 values, positions, K).
- G4 [inferred] The Phase 2 route (coefficients from the weights) would need P(err | V) to be predictable from the
  weights to a precision that a whole-formula refit on 1M samples just reaches. Given G3, even a perfect weight-
  derived P(err | V) would leave the estimator's variance at ~70% of sampling if used as a control variate. As a
  pure deductive estimate it would need the model-specific constants that G2 shows matter at the pp level.

## Alternatives not ruled out
- Richer V (m4, the rest of the order statistics, ages beyond the per-case blocks, T241 comparator-gain terms)
  might raise R² well above 0.3. The ceiling E[Var(err | V)]/Var(err) was not measured nonparametrically.
- The transfer test fits the shape on one model or a pool. A shape *parameterised by the weights* (Phase 2) is
  exactly what was not tried.
- The noisy-copy truths are at 2e7 samples (s.e. ≤ 1e-4), fine at the pp scale of the failures.

## Failure modes
- Logistic refit with a large fixed offset can diverge (V1 noisy0.01 K=3).
- Memory: designs above 1M × 30 needed chunking under the 2.5 GB cap; V3's LOO JSON was lost to an OOM.
