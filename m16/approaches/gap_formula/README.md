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
*Filled in after the runs.*

## Status
Pre-registered; running.

## Results
## Compute accounting
## Claims
## Alternatives not ruled out
## Failure modes
