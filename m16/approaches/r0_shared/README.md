# r0_shared (T300): R0's boundary shared across the max's position, aimed at the LENIENT bar on retrains only

↑ [r0_integral (T297)](../r0_integral/README.md) · [TASKS](../../../TASKS.md)

**Scope, declared post hoc (after T296):** the lenient bar, on retrains (DEV2) only. R0 is known to fail across
noise levels (T296), so noisy copies are out of scope here. This scope was chosen after seeing T296's results and
is labelled as such.

## Question
T297: R0's error boundary moves with v3 and g12 within a triple, so per-triple thresholds fail, and full 3-D
resolution per triple costs >= 20k FPE. Sharing the boundary across p1 (the max's position) leaves 90 (p2, p3)
classes and ~40 evaluations per class within 4k FPE. Is that sharing accurate enough?

## Pre-registration (written 2026-10-18, before any run)
**Stage 1 (diagnostic, sampling).**
- Per-triple R0 error mass E(t), for all 720 triples, on the 10 DEV2 models of T297 (indices 0, 8, ..., 72).
  Stratified MC: the same 20,000 value draws (v1 > v2 > v3 from the exact top-3 law) for every triple, so
  differences between triples have low variance.
- Sharing error of a rule = mean over the 90 classes of [E(p1_rep, p2, p3) - mean over p1 of E(p1, p2, p3)]. This
  is the error an estimator that integrates the shared boundary *exactly* would make.
- Rules, fixed now:
  - **S-mid**: p1_rep = the lower median of the 8 free positions;
  - **S-two**: the average of the two quartile positions (the 3rd and 6th of the 8 free positions; costs 2x
    evaluations);
  - **S-orc**: the best single p1 per class and model (an oracle lower bound, not a method).
- **Stop** if the sharing error of both S-mid and S-two exceeds .8 pp RMS over the 10 models.
- Stage 2 uses whichever of S-mid / S-two has the lower RMS. That selection on 10 of the 80 models is mild
  optimism and is stated as such.

**Stage 2 (estimator).** Per class: a deterministic 3-D boundary resolution of R0 at p1_rep, with exact masses and no
random nodes (design fixed after Stage 1, before any Stage 2 scoring). Scored on all 80 DEV2 models against the 1e8
truths.

**Go line (DRI):** lenient point F <= 10 at <= 4k FPE on DEV2, with F = RMS^2 x C_eff / mean p(1-p). With mean
p(1-p) = .0725 and C = 4k, that means total RMS <~ 1.3 pp. On a go: freeze est(model, budget) -> (estimate,
cost_flops) and send it to the T281 scorer for HP2.
