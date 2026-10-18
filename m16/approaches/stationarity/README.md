# stationarity — T310: training-stationarity identities as deductive family knowledge (Fable r3 B4)

↑ [TASKS](../../../TASKS.md) · review/fable_review_r3 (B4, branch claude/project-handoff-priorities-zpzfm8)

**All claims unverified. Stop checks use diagnostic sampling only.**

## Question
Every retrain is trained to a near-stationary point of the same expected loss, under AlgZoo defaults: AdamW, weight decay
0.001, batch 1024, x ~ N(0, I). So the expected readout gradient G = E[(softmax(L) - e_y) h_T^T] (10 x 16) is ≈ 0. Contracting
with W_o gives the calibration identity D = <G, W_o> = E[<p, L>] - E[L_y] ≈ 0.

Does such an identity pin an accuracy-adjacent quantity, deductively?

## Pre-registration (written 2026-10-18, before any T310 number was computed)
**Measurements** (80 DEV2 models; 2^22 = 4.19M N(0, I) inputs per model, rng 310, in chunks; diagnostic):
1. **Stationarity of the readout:**
   - ||G||_F and its sampling SE (batch-of-1024 means, bootstrap over 4096 batches);
   - the single-batch gradient norm s = mean over batches of ||G_batch||_F, batch 1024 = the training batch;
   - the residual ratio rho = ||G||_F / s.
   - Context: the AdamW stationary prediction ||G|| ≈ wd·||W_o ⊙ sqrt(v)||, with sqrt(v) approximated by the per-entry RMS of
     the batch gradient.
2. **Identity quantity:** D = E[<p, L>] - E[L_y] (identical to <G, W_o>), plus the expected loss E[-log p_y].
   - Across the 80 models report corr(D, acc) and corr(E[loss], acc), with acc = the open DEV2 1e8-sample truths (163cd62e).
   - Also corr(D_norm, acc), with D_norm = D / sd(L_y) (scale-free).

**Stop (DRI):** stop if the median rho > 0.10, OR corr(D, acc) < 0.5 in absolute value (the identity quantity must track
accuracy).
- corr(E[loss], acc) is reported but does not decide: the loss is not pinned by the identity.
- On a go: sketch, labelled as training-family knowledge, how a deductive estimator would use the identity. Cap 1.5 CPU-h.

**Alternatives written first.**
- (a) The identity holds to sampling precision for all models, so D ≈ 0 carries no model-to-model information (corr ~ 0).
  Stationarity fixes the identity's value; it cannot vary with accuracy.
- (b) Weight decay makes G ∝ -wd·W_o·sqrt(v), not 0. Then D ≈ -wd·<W_o, W_o ⊙ sqrt(v)> is a weight-only quantity: deductive,
  but likely a proxy for weight norm, not accuracy.
- (c) Residual non-stationarity (the LR floor at 0.01x) varies by kernel (fused vs torch), so any correlation is a kernel effect.
  Check: the correlation within each kernel.
