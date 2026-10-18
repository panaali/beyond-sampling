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

## Status
Done: **STOP** under the pre-registered rule (unverified), on the stationarity check.

## Results (unverified; `python m16/approaches/stationarity/stat.py /tmp/claude-0/dev2/weights.npz /tmp/claude-0/dev2/truths.json` -> stat.json, summary.json; `linesearch.py ... fused_s7`)
**Stationarity of the readout:**
- Residual ratio rho = ||E G||/E||G_batch||: median **0.41** (range 0.17–0.80; fused 0.44, torch 0.40).
- ||E G|| is 11–60x its sampling SE (median 28x).
- ||E G|| is ~280x the AdamW weight-decay stationary prediction wd·||W_o ⊙ sqrt(v)||.
- **The retrains are not stationary in W_o.**

**The gradient is real.** On fused_s7, with 2^20 fresh inputs and an autograd cross-check (diff 4e-16):
- the expected loss along W_o - eta·G falls 0.3030 -> 0.2997 at eta = 0.03;
- accuracy rises +0.14 pp.

**Identity quantity:**
- D = E<p,L> - E[L_y] = <G, W_o> (checked equal) ranges -0.071 … -0.010; its SE is 0.0003. So it is clearly nonzero.
- corr(D, acc) = **-0.52** (fused -0.47, torch -0.57).
- corr(E[loss], acc) = -0.991 (decides nothing: the loss is not pinned by any identity).
- corr(||G||, acc) = -0.49.

## Claims
- [measured] **Stop: median rho = 0.41 > 0.10.** The zoo-default retrains, with LR decayed to 0.01x, sit far from the readout
  stationary point. Near-stationarity identities therefore do not hold "for free": E[(p - e_y) h^T] is about 40% of a
  single-batch gradient.
- [measured] The calibration gap D is nonzero for every model and correlates only modestly with accuracy (|corr| 0.52 overall,
  0.47 within fused, below the 0.5 bar there). The corr bar passes overall only marginally.
- [inferred] A deductive estimator cannot use "G ≈ 0" as family knowledge. Its residual is the size of a batch gradient and
  varies by model. What D does carry is a property of the trained weights, and computing it needs the same expectation over
  inputs as accuracy, i.e. sampling or a closure.
- [inferred] The non-stationarity is consistent with Adam's sharp-direction oscillation at the final LR: the snapshot sits
  off-centre, so E[g] ∝ curvature x offset. The +0.14 pp gain along -G shows it is not pure noise.

## Alternatives not ruled out
- Only the readout gradient was measured; recurrent-weight identities were not.
- An identity averaged over the optimiser trajectory (e.g. the time-average of g ≈ 0) could hold even when the final snapshot is
  off-centre. It is not available from the final weights.
- Exhaustion: readout identities only. The full 432-parameter gradient and the second-order (Hessian) identities were not tried.

## Failure modes
Snapshot non-stationarity: Adam with a small final LR does not produce first-order stationarity at the saved point.

## Compute accounting
80 models x 4.19M forward passes (diagnostic), about 20 CPU-min; line search ~1 CPU-min.
