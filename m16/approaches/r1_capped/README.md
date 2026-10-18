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
