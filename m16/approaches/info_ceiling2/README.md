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
**Done. Verdict: INFORMATION-INFEASIBLE (scoped).** The best held-out RMS with <= 4k FPE of features (here 2.4k) is
.855 pp, against the > .45 line. Unverified; diagnostic and inductive by construction.

## Results
Reproduce: `PYTHONPATH=<alg_zoo shim> python3 perclass.py` (85 models, ~3 CPU-min) and `python3 learn.py`
(`results/learn.json`, `logs/learn.log`). Feature cost: 2,417 FPE per model (T300's estimator 1,697 + 720 median
margins).

| family | learner | inner-CV RMS (train 40) | **held-out RMS (45)** | torch (40) | seeds (5) | train fit | held-out mean err |
|---|---|---|---|---|---|---|---|
| **A (2.4k FPE)** | ridge, 3,600-dim per-class vectors | 1.725 | 1.767 | 1.862 | .593 | .331 | −.11 |
| A | lasso, same | 1.488 | 2.376 | 2.508 | .687 | .523 | −.24 |
| A | GBT, 24 aggregates | 1.101 | .942 | .991 | .374 | .226 | −.16 |
| A | **per-class basis ridge (selected)** | **1.059** | **.855** | .874 | .677 | .719 | +.07 |
| B (+probe-v1, ~6.3k) | ridge full | 1.721 | 1.763 | 1.858 | .592 | .330 | −.11 |
| B | lasso full | 1.107 | .813 | .830 | .668 | .526 | +.08 |
| B | GBT aggregates | .951 | **.723** | .761 | .258 | .061 | −.08 |
| B | per-class basis ridge (selected) | .814 | .980 | 1.016 | .618 | .438 | +.16 |

For reference, the T300 estimate used raw has held-out RMS 1.54. T308's 5-parameter linear fit on cheap summaries
gave .72, and with rank_exact .477.

**Readings:**
- **Nothing in the <= 4k family comes near .43–.45.** The best is .855 (selected) or .942 (GBT). Adding probe-v1
  (6.3k) brings the best to .72, the same as T308's linear fit. Richer learners do not extract more than the
  5-parameter summaries.
- **The high-dimensional learners overfit** (alternative 1): train RMS .33–.52 vs held-out 1.8–2.4. Forty training
  models cannot support 3,600 dimensions. The low-dimensional per-class basis and GBT do best, as expected.
- **The fused → torch shift is not the limit** (alternative 2). The torch RMS (.87) and inner-CV RMS on fused models
  (1.06) are comparable; the 5 seeds are easier (.26–.68).

## Claims
- C1 [measured] With per-class anchor quantities costing ~2.4k FPE (T300's thresholds, tie margins and tangents,
  median margins), no learner tried (ridge or lasso on 3,600 dims, GBT on aggregates, per-class basis ridge) predicts
  held-out accuracy better than .855 pp RMS. Nested selection, 45 held-out models.
- C2 [measured] Adding probe-v1 (~6.3k FPE) gives at best .72 pp, matching T308.
- C3 [inferred, scoped] For evaluation-based designs that see a model only through these anchor quantities at <= 4k
  FPE, the strict bar (~.43 pp at 4k) is **information-infeasible**: the information ceiling found here is about 2x
  the needed error. This covers these feature families and learners, not every possible cheap design (e.g. anchors
  at other skeletons, or other per-class probes).

## Alternatives not ruled out
- Other cheap per-class probes (different anchor skeletons, a second tangent direction) might carry information
  that these do not. Untested.
- With 40 training models, a learner that needs more data than that cannot be judged here. A larger panel (e.g. HP2's
  100) could raise the ceiling for high-dimensional learners.

## Compute accounting
Features: ~2.4k FPE × 85 models (~3 CPU-min). Learners with nested CV: ~5 CPU-min.
