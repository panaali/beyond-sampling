# T314: strict-question information ceiling — can richer fits of cheap per-class quantities reach <= .43 pp?

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [T308](../info_ceiling/README.md)

## Question
T308: a 5-parameter linear fit on cheap summaries (<= ~6.4k FPE) reached .72 pp held-out. Strict needs about .43 pp
at 4k FPE. Do richer fitted functions of the cheap per-class quantities reach that?
**Diagnostic and inductive by construction.**

## Pre-registration (2026-10-18, before any fit)
**Cheap feature budget: <= 4k FPE per model.** The per-class quantities at T300's model-free anchor (as in T308):
- from T300's estimator itself (`tb.est(n23=1, newton=1, nblind=0, use12=False)`, ~1.75k FPE): per class m_tie,
  dm_tie (margin and g23 tangent at the tie), tau (Newton threshold) and M (correct mass), plus the scalar T300
  estimate;
- m_med (margin at the median g23; 720 FPE).

The total is about 2.5k FPE. **Excluded:** probe-v1 (3.9k; T308's best cheap feature, but together with the rest it is
over 4k), the J tangents (5k) and rank_exact (25.8k). A secondary family B adds probe-v1 to the T300 parts (≈ 5.7k FPE)
and is reported, but not used for the verdict.

**Data:** 80 DEV2 models (truths 1e8) + 5 LB3 seeds (32M). Train = fused_s0..39, test = torch_s0..39 + seeds. All
hyperparameters by **inner 5-fold CV on the 40 training models only** (nested).

**Learners (fixed now):**
1. **Ridge** on the full per-class vectors: [m_tie, dm_tie, tau, M, m_med] × 720 = 3,600 standardized dims, plus the
   T300 scalar. Alpha from a grid of 13 log-spaced values (1e-3 .. 1e9).
2. **Lasso** on the same, alpha grid of 20 log-spaced values.
3. **Gradient-boosted trees** (sklearn, depth 2, learning rate .05, n_estimators in {50, 100, 200, 400},
   subsample .8, random_state 0) on 24 class-aggregated features:
   - mean Φ(−m_med/s) and mean Φ(−m_tie/s) for s in {.25, .5, 1, 2, 4, 8};
   - mean min(tau, 3) and the fraction tau = 0;
   - quantiles (10/50/90%) of m_med and of dm_tie;
   - mean M; the T300 estimate.
4. **Per-class error-mass regression:** acc = β0 + Σ_k β_k · mean_c b_k(class c). Here b_k is a shared per-class basis:
   - Φ(−m_med/s) and Φ(−m_tie/s) at the 6 scales;
   - 1[tau = 0] and min(tau, 3);
   - M;
   - the products Φ(−m_med/1) · min(tau, 3) and Φ(−m_tie/1) · M.

   That is 19 basis means, fitted with ridge (nested alpha).

The target is accuracy in pp.

**Verdict (DRI):**
- best held-out RMS (45 test models) **> .45 pp** with <= 4k FPE features → the cheap strict regime is
  **information-infeasible** for evaluation-based designs with these feature families (scoped);
- **<= .43 pp** → report which features carry the signal;
- in between: neither.

The best learner is chosen by inner-CV RMS on the training models, not by test RMS. All four test RMS values are
reported anyway.

**Alternatives first.**
1. With 40 training models and 3,600 dims, high-dimensional learners cannot beat the low-dimensional summaries
   (variance), whatever information is present. The per-class regression (4) is the main guard against that.
2. The fused → torch shift could cap held-out accuracy below the information ceiling. Check the seeds and train-CV
   separately.

## Status
Pre-registered.
