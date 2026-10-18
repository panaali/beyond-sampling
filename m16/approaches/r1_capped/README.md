# r1_capped (T315): capped exact walk (c gate cells) + declared linear extrapolation, all classes, executed B2

↑ [r1_walk (V1)](../r1_walk/README.md) · [r0_shared (T300)](../r0_shared/README.md) · [r0_walk (T302)](../r0_walk/README.md) ·
[TASKS](../../../TASKS.md)

**Scope: lenient bar, retrains. The claim is HP2 only. DEV2 is development data.**

## Pre-registration (2026-10-18, before any run)
**Estimator "CAP-c".** R1W V1 in every respect (closure c: v4 at the latest free position with its 1-node
conditional value and 6 lows at mu(v4); one conditional-mean transverse point per region; tmax = .5; exact
model-free masses), except for three changes:
1. **Capped walk.** On each line, walk exactly through the first **c** gate cells (c in {1, 2, 3}). From the end of
   the c-th cell to tmax, the margin is extended with that cell's exact affine form. The correct set on that piece
   is solved in closed form. **Declared approximation:** no further gate change within the walk range. It is not an
   inductive step: the extended form is the network's own exact linear map on the last walked cell.
2. **All four lines** in both regions: nonblind g23 and g12, and blind g23 and g12. V1 assumed blind g12 correct;
   here it is walked. No error class is omitted.
3. **B2 executed.** In the first cell every row sits at the same walk parameter. The first cell is evaluated as a
   prefix tree: each distinct (input, tangent) prefix across the 720 triples is computed once, and the gate-flip
   root search is shared along the tree. It is charged as executed.

**Cost:** the Directive 8 convention, instrumented as in T313:
- 1,218 FLOPs per step-evaluation;
- 790 per cell readout and correct-set solve, and the same for the extrapolated piece;
- Phi at weight-dependent points 20 (+3 for ^8);
- model-free tables free.

**Selection:** c in {1, 2, 3} with the lowest point F on DEV2 (all 80, 1e8 truths, own charged cost).
**Go (DEV2, development):** point F <= 10 **and** upper 95% CI <= 12. Then freeze and send to the T281 scorer.
**The claim is HP2 alone.** Per-class error split: estimator vs MC of the closure, nonblind / blind, on the 10 T297
models.

**Forking paths:** this would be the third R1W-family HP2 submission (T300 stage 0, T306 V1; T313 was a cost
re-count, not a submission). Across T297-T315, about 33 variants have seen DEV2.

## Results (`python f80.py`, `results.json`; all 80 DEV2, development)
**Validation.** Uncapped (c = 400), CAP reproduces the exact walk `r1w.est(K=1, blind12=True)` to **0.0** (fused_s8).
So the executed prefix tree (B2) and the bookkeeping are exact; only the cap changes anything.

| c | C (FPE at 5,948) | bias (pp) | RMS (pp) | point F [95% CI] |
|---|---|---|---|---|
| 1 | 4,343 | -10.44 | 14.56 | 1,269 [863, 1,707] |
| 2 | 7,317 | -8.96 | 12.90 | 1,678 [1,132, 2,223] |
| 3 | 9,818 | -5.30 | 8.17 | 903 [593, 1,230] |

- **FAIL**, far from the go line. The pre-registered selection would pick c = 3 (F 903), so nothing is sent to HP2.
- **Per-region diagnosis** (correct-mass difference against the uncapped walk):

  | model | c | nonblind | blind |
  |---|---|---|---|
  | fused_s8 | 1 | -3.93 pp | -0.01 pp |
  | fused_s8 | 3 | -2.69 pp | -0.02 pp |
  | torch_s24 | 1 | -1.86 pp | -0.04 pp |
  | torch_s24 | 3 | -0.18 pp | -0.00 pp |

  The failure is entirely in the nonblind lines, and on other models it is much larger (RMS 8-15 pp).
- **Why:** near the tie the first gate cells are tiny (H-E15's tie fragmentation). The affine margin of an early
  cell, extended over the 0.5 walk range, crosses into the wrong class, while the true margin, after its later gate
  flips, does not. "No further gate change" is a bad declared approximation exactly where the walk starts. The
  error is systematically one-signed (bias always negative), so it is not noise.
- Contrast with T300's Newton threshold, which also extrapolates from the first cell. Its sign convention (tau = 0
  when the margin is positive at the tie, clipped at 3) avoided this collapse but leaned on cancellation. Here,
  with all classes, the plain extrapolation fails outright.

## Status
Done (2026-10-18). FAIL, no HP2 submission (the family stays at two HP2 submissions). Forking paths: 3 more
variants (c = 1, 2, 3) have now seen DEV2, about 36 in total.

## Compute accounting
3 variants x 80 models, plus the validation: ~30 CPU-min.
