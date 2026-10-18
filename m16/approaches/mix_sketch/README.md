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
