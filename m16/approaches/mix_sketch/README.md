# mix_sketch: T309, kill test for future-aware mixture propagation (Fable r3 B3), plus B4's cheap stop checks

↑ [Approaches index](../README.md) · [TASKS](../../../TASKS.md) · design: branch claude/project-handoff-priorities-zpzfm8, review/fable_review_r3/README.md B3, B4

## Pre-registration (written 2026-10-18, before any T309 number)
**Question (B3).** If the state law at step 8 is represented as a Gaussian mixture whose components are merged only
along directions the future readout ignores, how many components per class are needed to reproduce the per-unit
P(correct) within 10%? **Stop: > 50 per class** (then B >= 36k components, cost >= 1e5 FPE, and the method is dead in
the cheap regime, scoped).

**Oracle design (diagnostic sampling; gives the method every advantage).**
- State at step 8 = z = (h_8, x_8, x_9): the hidden state after inputs x_0..x_7, plus the two remaining scalar
  innovations (18-D), sampled exactly from the unit's law (1e5 draws).
- A K-component mixture is built by clustering the samples (deterministic k-means++, seed 309, 15 Lloyd iterations).
  Each component is the Gaussian with that cluster's exact sampled mean and covariance, weighted by the cluster's mass.
- The **push-forward of each component through steps 8-9 is exact**: I draw from the Gaussian and run the true
  network. This is at least as good as B3's closed-form "17 truncated pieces" step, which is exact only for point-mass h.
  So the failure of the oracle implies failure of the closed form, up to lucky cancellation. P_hat = sum_k w_k
  P(correct | component k).
- Merge rules (which directions keep components apart):
  - **R-naive:** k-means on the whitened z.
  - **R-future (B3's rule):** k-means on f = A z (9 rival margins). A is the backward adjoint of W_o through the
    unit's modal late gate pattern (the most common gate pattern at steps 8-9 among the unit's samples). So
    components differ only along directions the readout reads.
  - **R-future+gates:** f plus the 16 step-8 pre-activations u_8 = W_h h_8 + W_i x_8, each block standardised. This
    lets the clusters also separate where the likely late gates differ.
- K ladder: 1, 2, 4, 8, 16, 32, 50, 64, 128. Per unit, K* = the smallest K on the ladder that passes, or "> 128".
- Reference P: 4e5 separate exact samples. Push-forward: 2e5 draws allocated by weight.

**Units.** 5 DEV2 models (#0, #5, #20, #40, #60). Per model:
- 10 blind units (T293's (case, label) units, my own draw with rng 309);
- 10 non-blind classes (ordered top-3 position triples, drawn with rng 309; law N(0,I) given that top-3 ordering and
  x_(2) > 0).
That is 100 units in all.

**Criterion "within 10%":** |P_hat - P| <= 0.10 min(P, 1-P), or <= 0.005 absolute. Relative to min(P, 1-P), so
that a non-blind class with P ~ .95 is held to 10% of its error rate.

**Verdict:** using the best rule, if the median over units of K* is > 50 (counting "> 128" as large), STOP: dead in
the cheap regime (scoped to mixture-at-step-8 representations with these merge rules). If median K* <= 50 and >= 80%
of units pass at K <= 50, it is a go signal for building the closed-form step. Otherwise it is inconclusive. I also
report the blind and non-blind halves separately.

**B4 cheap stop checks (if time remains in the 1.5 h cap)**, on all 80 DEV2 models (diagnostic sampling):
- (1) Readout-gradient stationarity: G = E[(softmax(L) - onehot(y)) h_10^T] from 1e6 samples. Report
  |G|_F / |g_B|_F, where g_B is a single batch's gradient (batch B = 1024, measured as the sampling SD of the batch
  gradient). Stop if > 10%.
- (2) corr over the 80 models of (E[<p, L>] - E[L_y]) with accuracy, p = softmax(L). Stop if |corr| < .5.
Labels: everything here is diagnostic sampling. Nothing is an estimator.
- **Amendment (2026-10-18, before any panel unit was run).** It was written after I looked at one development unit
  that is not a panel unit by construction: DEV2 #0, class (3,7,1). There every rule was still at .82-.87 against
  P = .956 with 128 components.
  - (i) **Cap:** 50 units (the first 5 of each pre-registered draw per model: 25 blind + 25 classes), with 6e4 state
    samples, 2e5 reference and 1.2e5 push-forward draws. At ~55 s per unit the 100-unit plan does not fit 1.5 h.
  - (ii) **Two fairer rules added.** They draw the innovations x_8, x_9 exactly: resampled from the component's own
    draws, with h_8 from the component's Gaussian regression on them, as B3's "scalar-innovation-exact" step requires.
    - `future_gates_X` (state h_8);
    - `pre_future_gates_X` (state = the step-7 pre-activation u_7, with h_8 = ReLU(u_7) applied exactly; this is the
      truncated-pieces form before any re-merge).
    The verdict uses the best of all five rules per unit (most generous). The rules `naive`, `future` and
    `future_gates` (Gaussian over all 18 dims) are kept.
- **Scope note (DRI, 2026-10-18, before B4 ran):** the B4 checks moved to T310 in another child; not run here.

## Method
`mix.py` (own code; diagnostic sampling): the unit samplers (blind units from `../mf_verify/mfv.py`, which is
exact; non-blind classes by permutation of sorted draws, checked). It computes the modal late gate pattern and the
adjoint A, a numpy k-means++, the five merge/representation rules, and the exact push-forward. `summarize.py` ->
`results/mix_summary.json`. Raw per-unit, per-rule, per-K P_hat: `results/mix.json`.
Run: `python mix.py && python summarize.py` (about 49 CPU-min, single process, < 1 GB).

## Status
**Done: STOP** (pre-registered). Dead in the cheap regime, scoped to step-8 Gaussian-mixture state representations
with readout-aware (adjoint) or naive merging, even with an exact push-forward through the last two steps.

## Results (50 units: 25 blind + 25 non-blind classes on DEV2 #0/#5/#20/#40/#60)
K* = the smallest ladder K that passes the 10% criterion. "best" = the best of the 5 rules per unit (most generous).

| | blind (25) | non-blind classes (25) | all (50) |
|---|---|---|---|
| median K*, best rule | 16 | **> 128** | **> 128** |
| pass at K <= 50, best rule | 56% | **4%** | 30% |
| pass at K = 1, best rule | 28% | 0% | 14% |
| post hoc "stable" K* (smallest K from which every larger rung passes), median | 50 | > 128 | > 128 |
| median relative error (in min(P,1-P)) of the best rule at K = 1 / 8 / 50 / 128 | .31 / .15 / .10 / .06 | **4.0 / 2.5 / 1.7 / 1.4** | 1.2 / .81 / .49 / .38 |

Per rule, median K* is > 128 for every single rule on every half. Pass at K <= 50 is .32-.40 on blind units and .00-.04
on classes. The readout-aware merge (future, future_gates) is **no better than naive k-means**. Exact innovations
(the _X rules) help by a few points. The pre-activation form (ReLU at step 8 exact) does not help.

Signs [measured]: on the classes the mixture underestimates P(correct) on 25/25 units (median -.068 at K = 128,
future_gates_X), against an error rate of only ~.05. Blind units: 80% under. The modal late gate pattern covers a median
72% of a unit's draws, so the "likely late gates" adjoint misreads ~1/4 of the mass.

## Claims
1. [measured] A step-8 Gaussian-mixture state, even with exactly sampled component moments, exact innovations and an
   exact push-forward through the last two steps, does not reproduce per-class P(correct) within 10% of the error
   rate with <= 128 components on 24/25 non-blind classes. The median relative error is still 140% at K = 128.
   By the pre-registered rule: **STOP**, B >= 720 x 50 = 36k components, cost >= 1e5 FPE.
2. [measured] Merging along the readout adjoint (ARC's sketching rule as Fable B3 states it) gives no gain over
   naive merging here.
3. [inferred] The error rate of a non-blind class (~5%) lives in thin regions of the state law: near-tie
   configurations whose h_8 sits near the late-gate walls. Any Gaussian component smears them, and the error
   decays only like ~K^-0.2 (4.0 -> 1.4 over K = 1 -> 128). [speculative] Extrapolating, reaching 10% would need
   K ~ 1e7 per class.
4. [measured] Blind units are easier (P is mid-range, so 10% is a looser absolute bar). Even there only 56% pass at
   K <= 50.

## Alternatives not ruled out
- Mixtures fitted by EM, or mixtures of *truncated* (rectified) Gaussians at step 8, rather than k-means Gaussians.
  The truncated-piece closed form B3 describes would keep the rectification of h_8. My `pre_` rule keeps the last ReLU
  exact and did not help, so I do not expect a large gain.
- Representation at an earlier step (e.g. step 9, one exact step) was not tried. Step 8 was the brief's choice.
- Clustering on the full late gate pattern (16 x 2 bits) as a categorical split was not tried. It is T295's branch
  route, where N99 grows 47 -> 698 per step.
- 50 units, not 100 (cap). The non-blind result (24/25 fail at K <= 128) leaves little room for unit-sampling noise.

## Failure modes
- Readout-adjoint merging needs one "likely" gate pattern. Units whose late gates vary (modal share ~.3-.7) have no
  single adjoint, and the merge direction is wrong for the rest of the mass.

## Compute accounting
About 49 CPU-min (50 units x 5 rules x 9 K values; reference 2e5, state 6e4, push-forward 1.2e5 draws per (rule, K)).
All diagnostic sampling.
