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

## Results: Stage 1 — **STOP** (pre-registered) (`python stage1.py 20000`, `results/stage1.json`)
Per-triple R0 error masses from 20,000 common value draws per triple (14.4M R0 forwards per model, 10 DEV2 models):

| model | R0 error | sharing error S-mid (pp) | S-two (pp) | S-orc (oracle) (pp) | within-class s.d. of E over p1 (pp) |
|---|---|---|---|---|---|
| fused_s0 | 11.6% | -3.07 | -3.14 | -0.38 | 10.0 |
| fused_s8 | 7.6% | -0.99 | -0.53 | -0.09 | 4.8 |
| fused_s16 | 9.5% | -2.20 | -2.02 | -0.30 | 8.4 |
| fused_s24 | 5.4% | -1.03 | -0.61 | -0.41 | 4.6 |
| fused_s32 | 5.1% | -0.91 | -0.55 | -0.21 | 3.5 |
| torch_s0 | 4.6% | -0.22 | -0.32 | -0.19 | 3.2 |
| torch_s8 | 6.0% | -0.45 | -1.03 | -0.27 | 4.3 |
| torch_s16 | 10.4% | -2.57 | -1.25 | -0.64 | 8.2 |
| torch_s24 | 11.7% | -2.97 | -1.40 | -0.74 | 9.0 |
| torch_s32 | 7.9% | -1.53 | -0.10 | -0.04 | 5.6 |
| **RMS** | | **1.87** | **1.40** | 0.39 | |

- **Both rules exceed .8 pp, so the stage stops.** Stage 2 was not built.
- The max's position matters too much to share. Within a (p2, p3) class, R0's per-triple error mass varies by
  3-10 pp (s.d. across the 8 p1 values). Per-class absolute sharing errors are 1.5-6.5 pp.
- The sign is consistently negative: the middle/quartile representatives have *less* error than the class mean.
  The class mean is pulled up by extreme max positions, consistent with the clock: how long ago the max arrived
  matters.
- Even the oracle (the best single p1 per class and model, chosen with the answer) leaves 0.39 pp, close to half
  the stop line before any integration error.
- [inferred] A shared-boundary design would have to share along a coordinate that R0's error does not depend on.
  p1 is not one. Sharing across p3 or the low positions was not tested.

## Status
Done (2026-10-18). Stopped at Stage 1 by the pre-registered rule. Nothing was sent to the T281 scorer.

## Compute accounting
Stage 1: 10 models x 720 triples x 20,000 R0 forwards = 1.44e8 forwards (diagnostic sampling), ~12 CPU-min.

## Stage 0 (DRI request received after Stage 1 ran; pre-registered here before running it)
- Score the four existing T297 designs (r0_integral/tb.py, frozen; no change of any constant) against the DEV2 1e8
  truths on **all 80 retrains**.
- Report total bias, RMS, and point F = RMS^2 x C / mean p(1-p) at each design's own charged cost C (FPE; T196
  FLOPs as implemented). This is my reading of "amendment 2": cost as charged, no budget rounding.
- **Lenient pass:** point F <= 10 on DEV2. The best passing design is frozen and sent to the T281 scorer.
- Caveats, stated in advance:
  - (a) the designs were built while looking at 10 of these 80 models (T297 chose nothing by fitting, but the
    variant ladder was seen);
  - (b) DEV2 is an open panel, so a pass is a DEV2 statement and HP2 decides;
  - (c) scope is retrains only, declared post hoc after T296.
