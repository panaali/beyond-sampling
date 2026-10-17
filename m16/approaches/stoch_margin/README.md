# stoch_margin (T1): stochastic reduction of the tie margin

↑ [Approaches index](../README.md) · [INDEX (registry + ledger)](../../../INDEX.md) · [TASKS](../../../TASKS.md)

## Question
Every deterministic summary of the low inputs overestimates accuracy (H-E10). Does it help to carry the low inputs
as a *distribution* instead? Condition on a skeleton S (positions and values of the top 2-3 inputs). The other
inputs are then iid N(0,1) truncated below the 3rd value. Compute the law of the residual tie margin
deterministically (convolution, cumulants or a small gate-pattern mixture), take q(S) = P(correct | S), and
integrate q over the low-dimensional skeleton space with deterministic quadrature. Can this beat sampling on
leaderboard v3?

## Pre-registered alternatives (written before the full diagnostic; AGENTS.md rule 3)
The reduction needs q(S) correct to about 0.1 pp in aggregate. Possible reasons it could fail, with the cheapest
test that separates them:
1. **Non-additivity (gates).** The low inputs interact through ReLU gates, so the margin given S is not a sum of
   independent per-input terms. Test: a paired cut-HDMR additive surrogate (each 1-D term exact) against the model
   on the same low draws.
2. **Non-Gaussian shape, not interactions.** The margin given S may have the right first two moments but a shape
   (skew, atoms, multimodality from gate switches) that a 2-moment closure misses. Test: a Gaussian with the TRUE
   conditional mean and covariance of the 9 margins (a ceiling for any 2-moment method).
3. **Skeleton too coarse.** The error may be driven by one or two low inputs (the 4th largest sits near v3), so no
   noise model of the low inputs is accurate unless they enter the skeleton. Test: repeat 1-2 with a top-4 and a
   top-5 skeleton.
4. **Cost, not fidelity.** Even an accurate q(S) may cost too much: 720 position triples times a 3-D value
   integral, at roughly 10-20 forward-pass equivalents (FPE) per node, fits poorly into 16k-64k FPE. Test: count
   the nodes that a smooth-q quadrature needs per triple.

## Method
All tests are paired diagnostics with random inputs, **for diagnosis only**. No estimator was built (see Status).
- Skeleton S = positions and values of the top-K inputs, drawn from the exact law. Given S, the other 10-K inputs are
  iid N(0,1) | < v_K, which is exact. q(S) = P(correct | S) is estimated from M = 256 low draws per skeleton.
- Each reduction is evaluated **on the same low draws** as the model. The number reported is therefore the error of
  the reduction's *exact* law, which is what a perfect deterministic convolution or characteristic-function step
  would return, with no integration error added.
- Reductions tested:
  - R0: low inputs at the conditional mean (deterministic summary).
  - lin: exact linear map of the gate pattern at the conditional-mean point.
  - add: cut-HDMR additive surrogate L0 + sum_j [L(base with x_j) - L0], each 1-D term exact; this is the
    "sum of independent contributions" hypothesis.
  - gauss: a Gaussian with the TRUE conditional mean and covariance of the 9 margins; this is the ceiling for any
    2-moment closure.
  - lin_med: lin at the conditional median.
  - mixR / nearR: mixtures of linearisations at R anchors (a proxy for a gate-pattern mixture).
  - oracle-k: the k most frequent full gate patterns per skeleton computed exactly, the rest by lin. This is an
    upper bound for a small gate-pattern mixture with exact region masses.

| file | what |
|---|---|
| `sm.py` | helpers (models, skeleton sampling, truncated normals) |
| `diag_cond.py` | diagnostic 1: R0 / lin / add / gauss at K = 3 |
| `diag_condk.py` | diagnostic 1b: the same for a top-K skeleton (K = 2..5) |
| `diag_gates.py` | diagnostic 2: gate-pattern concentration given S, oracle mixture ceiling, share of x4, skew |
| `diag_fix.py` | diagnostic 3: anchor mixtures (the most promising fix at K = 3) |

## Status
**Done. Negative**, stopped after the diagnostics as AGENTS rule 3 and the T1 brief require. The reduction's
aggregate error is far above the ~0.1 pp bar at the skeleton sizes where it would help. At the skeleton size where
it is accurate (K = 5), it can buy at most a 1.2x variance reduction. No estimator was built and none was scored on
the leaderboard, because by the numbers below one could not beat sampling at 4k-64k FPE.

## Results
Commands are run from the repo root with `OMP_NUM_THREADS=1` and the venv active. Models: `orig` and the DEV copies
`dev_s0.0025_0` and `dev_s0.005_0` (test seeds were not used). "bias" = mean over skeletons of q_red - q, in pp of
accuracy (negative = the reduction predicts too many errors), +- s.e.; "abs" = mean |q_red - q| per skeleton.

**1. Top-3 skeleton (the T1 proposal)**: `python m16/approaches/stoch_margin/diag_cond.py MODEL 20000 256`

| model | R0 (mean point) | lin | add (independent sum) | gauss (true moments) |
|---|---|---|---|---|
| orig | -0.03 +- 0.08 (abs 2.9) | -0.45 +- 0.06 (abs 2.3) | **-3.79** +- 0.06 (abs 4.0) | **-2.21** +- 0.02 (abs 2.4) |
| dev_s0.0025_0 | +0.18 +- 0.08 (abs 3.2) | -0.34 +- 0.06 (abs 2.1) | -3.63 +- 0.06 (abs 3.9) | -2.20 +- 0.02 (abs 2.4) |
| dev_s0.005_0 | +0.84 +- 0.09 (abs 4.5) | -0.13 +- 0.06 (abs 2.4) | -3.45 +- 0.06 (abs 3.8) | -2.14 +- 0.02 (abs 2.4) |

**1b. Larger skeletons**: `python m16/approaches/stoch_margin/diag_condk.py MODEL 10000 256 K`

| K | model | R0 | lin | add | gauss |
|---|---|---|---|---|---|
| 4 | orig | +0.21 (abs 1.7) | -0.04 (abs 0.86) | -0.52 (abs 0.67) | -1.02 (abs 1.20) |
| 4 | dev_s0.005_0 | +0.53 (abs 3.1) | -0.11 (abs 0.99) | -0.42 (abs 0.68) | -1.03 (abs 1.28) |
| 5 | orig | +0.20 (abs 1.1) | +0.01 (abs 0.36) | **-0.05** +- 0.01 (abs 0.13) | -0.44 (abs 0.58) |
| 5 | dev_s0.005_0 | +0.42 (abs 2.3) | +0.01 (abs 0.46) | **-0.04** +- 0.01 (abs 0.17) | -0.44 (abs 0.70) |

(The K = 2 run crashed on a row-format bug and was not rerun: K = 3 already fails, and K = 2 puts the 2-3 tie itself
into the noise.)

**1c. Leverage: how much of the outcome variance the low inputs carry.** If q(S) were exact, integrating out the
low inputs could cut the variance at most by p(1-p) / Var_S(q) (the conditional-Monte-Carlo gain ceiling):

| K | orig | dev_s0.0025_0 | dev_s0.005_0 |
|---|---|---|---|
| 3 | 1.60 | 1.54 | 1.54 |
| 4 | 1.32 | - | 1.34 |
| 5 | 1.20 | - | 1.25 |

(Computed from the saved `data/stoch_margin/diag_cond*.npy`, with E[q(1-q)] unbiased over 256 draws; the one-liner is in the session log.)

**2. Gate patterns given S (K = 3)**: `python m16/approaches/stoch_margin/diag_gates.py MODEL 4000 256`
- Gate patterns are not concentrated:
  - distinct full gate patterns per 256 draws: median 148 (orig) and 151 (dev_s0.005_0);
  - share of draws in the top 1 / 5 / 20 / 50 patterns (error-weighted): 12 / 26 / 44 / 59% (orig) and
    15 / 33 / 52 / 66% (dev).
- Oracle mixture (top-k patterns exact, the rest by lin), bias for k = 1 / 5 / 20 / 50: -0.36 / -0.32 / -0.27 / -0.21 pp
  (orig) and -0.20 / -0.17 / -0.15 / -0.12 pp (dev). Even 50 exactly-known patterns per skeleton do not reach 0.1 pp.
- The largest low input x4 explains only 15% (orig) and 11% (dev) of the conditional error variance: the errors are
  driven collectively by all the low inputs. The median skew of the 2-vs-3 margin is about 0.

**3. Most promising fix at K = 3, anchor mixtures of linearisations**:
`python m16/approaches/stoch_margin/diag_fix.py MODEL 10000 256`

| model | lin_mean | lin_med | mix3 | mix5 | near3 | near5 |
|---|---|---|---|---|---|---|
| orig | -0.36 (abs 2.2) | -0.72 | -13.1 | -13.2 | -2.55 | -3.45 |
| dev_s0.005_0 | -0.18 (abs 2.6) | -0.48 | -11.7 | -12.2 | -2.14 | -3.17 |

A linearisation at an off-centre anchor extrapolates very badly to the rest of the distribution, so mixing anchors
makes things worse. The single centre linearisation remains the best K = 3 reduction (-0.13 to -0.45 pp across 3
models, with per-skeleton errors of 2-2.5 pp that happen to cancel partly).

**4. Cost argument (alternative 4)** [inferred from 1-3]:
- A deterministic estimator built on lin at K = 3 costs about 4-5 FPE per skeleton node: one forward pass, plus
  7 tangent passes from each low position (about 0.55 FPE each), plus the exact law of a linear form of truncated
  normals, which is model-free arithmetic.
- Integrating q over skeletons by plain conditional MC would give MSE / sampling of about c x Var_S(q) / p(1-p) =
  4.5 / 1.6, i.e. about 2.8x WORSE than sampling, before adding the -0.13 to -0.45 pp bias.
- A deterministic skeleton quadrature must therefore gain at least about 3x over MC on 720 position triples x 3-D
  value integrals, with about 5-20 nodes per triple at 16k-64k FPE. q(S) is step-like in the 2-3 gap (E[q(1-q)] is
  only 0.016-0.028). Previous face quadratures got 1-4x at T = 10 (E25).
- The bias alone exceeds sampling's RMSE at 64k (0.08 pp) on every model tested, and at 16k (0.17 pp) on 2 of 3.
- At K = 5 the bias would be acceptable, but there are 30,240 ordered position tuples x 5-D values, and the ceiling
  is 1.2x.

## Compute accounting
- Diagnostics only, all with random inputs. None of it is charged to any estimate.
- Forward-pass equivalents:
  - diag_cond: 3 models x 20k skeletons x 256 draws x (1 + 7 cut-HDMR) passes, about 1.2e8 (most on short
    partial-length-free batches);
  - diag_condk: about 1e8;
  - diag_gates: 2 x 4000 x 256, about 2e6;
  - diag_fix: 2 x 10k x 256 x (1 + 5 linearisations of 8 passes), about 2.5e8 batched rows.
- Total CPU about 25 min on one core.

## Claims
- [measured] Given the top-3 skeleton, the margin is **not** a sum of independent per-low-input contributions. The
  exact law of the cut-HDMR additive surrogate overstates the error by 3.5-3.8 pp (per-skeleton |error| about 4 pp),
  on orig and two DEV noisy copies.
- [measured] The conditional margin law is far from Gaussian: a Gaussian with the exact conditional mean and
  covariance of the 9 margins overstates the error by 2.1-2.2 pp. Every 2-moment closure fails by more than 20x the
  bar, even at this fine conditioning (720 triples x exact values).
- [measured] The best K = 3 reduction (exact linear map of the centre gate pattern, exact law) has an aggregate bias
  of -0.13 to -0.45 pp and per-skeleton errors of 2.1-2.6 pp. Small gate-pattern mixtures do not fix it: gate
  patterns given S are nearly all distinct (148 of 256), and an oracle with the 50 most frequent patterns exact
  still leaves -0.12 to -0.21 pp. Anchor mixtures make it worse (-2 to -13 pp).
- [measured] Fidelity improves with skeleton size: additive -3.8 -> -0.5 -> -0.05 pp for K = 3 / 4 / 5 (orig).
- [measured] Leverage falls with skeleton size: the low inputs carry at most 1.6x (K = 3), 1.3x (K = 4) and 1.2x
  (K = 5) of variance reduction (conditional-MC ceiling).
- [inferred] **Fidelity-leverage trade-off**: a distributional treatment of the low inputs is accurate only once
  so few inputs remain random that integrating them out buys almost nothing. The skeleton integral, where nearly all
  the variance sits, has a near-step integrand over hundreds to tens of thousands of discrete position tuples. So
  this reduction cannot beat sampling on M_16,10 at 4k-64k FPE.
- [inferred] H-E10 needs a qualifier. A deterministic summary at the conditional mean (R0) is *not* always biased
  upward: -0.03 +- 0.08 pp on orig at K = 3. It is upward on the noisy copies (+0.18, +0.84) and at K = 4-5
  (+0.2 to +0.5). Poincare's +0.25 pp at K = 3 used a different summary (expected order statistics in order). The
  Jensen effect is real, but its size depends on the summary, and per-skeleton errors are 3-4 pp in both directions.

## Alternatives not ruled out
- **Different noise coordinates.** Carry the hidden state (or the eigenmode/phase coordinates) as the random
  object instead of the raw low inputs, e.g. a sequential stochastic linearisation with re-linearisation at each step.
  The K = 3 Gaussian ceiling (-2.2 pp, already using the exact moments of the margins) suggests any Gaussian state
  closure fails too. A non-Gaussian sequential closure was not tested.
- **ANOVA rather than cut-HDMR additivity.** E26 found that an oracle ANOVA-additive fit misses the band integral by
  only 4%, against our cut-HDMR's about 80%. Its components are conditional expectations, so computing them needs
  integrating the other inputs; a deterministic route to them was not found.
- **Combination with exact tie-direction slicing.** Integrate along the 2-3 tie direction exactly, as in E25, and
  carry the 7 rest inputs by the lin law. This would make the face integrand 2-D per triple, but it inherits the lin
  bias (-0.13 to -0.45 pp, i.e. 3-10% of the error mass). Not built.
- The K = 2 diagnostic crashed and was not rerun; test seeds 0-4 were not examined, to keep them clean (orig and 2
  DEV copies only). Per-model numbers carry the stated s.e.; 256 draws per skeleton bias E[q(1-q)] slightly
  (corrected).

## Failure modes
- K <= 3: the gates switch with the low inputs (about 150 distinct patterns per 256 draws), so any closure that
  is additive, Gaussian, single-pattern or few-pattern is off by 0.1-4 pp.
- K >= 5: accurate, but the position-tuple count (30,240) and a step-like q(S) make deterministic integration of
  the skeleton costlier than sampling, and the analytic part gains at most 1.2x.
- Off-centre linearisations extrapolate badly (anchor mixtures, -12 pp).
