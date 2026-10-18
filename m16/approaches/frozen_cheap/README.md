# frozen_cheap — T287: can witness/frozen-gate class estimators be made cheap enough for F <= 10 (Directive 8 amendment 2)?

↑ [TASKS](../../../TASKS.md) · [frozen_check (T282)](../frozen_check/README.md) · T274b (lenient_bb)

**All claims unverified.**

## Pre-registration (written 2026-10-18, before any T287 number was computed)
**Score (DRI, Directive 8 amendment 2).** F = mean(e^2) x max(C, 1k FPE) / mean(p(1-p)) per LB3 group. Here e is the estimator's
error on each model, C its charged cost in FPE (1 FPE = 6000 FLOPs at (16,10); instrumented, every weight-dependent op), and
p the model's accuracy (leaderboard 3.2e7-sample truths). Strict F <= 1, lenient F <= 10.

**Estimator E1(design)** (deterministic): estimate = sum over classes c of w_c v_c.
- w_c: the class's exact model-free probability (combinatorics of i.i.d. N(0,1) orderings and sign counts; free model-free
  precomputation).
- x_c: a constructed witness. Model-free conditional means of the order statistics given the class's constraints, placed at the
  class's positions.
- v_c = 1 if the trained net (one forward pass of x_c) classifies x_c correctly, else 0.
- Cost: one forward pass per class (charged 1 FPE each), plus a negligible label/argmax.
- Labels (Hilton): w_c deductive; x_c constructed (allowed); v_c applied to the whole class is **inductive** (it extrapolates
  one input's verdict to the class). The frozen-gate "perfect integration" numbers below are the deductive ideal of the same
  class design (diagnostic, measured by paired sampling as in T282).

**Designs (refining classes):**
- D0 = (top-2 positions, k = number of positives): 90 x 11 = 990 classes.
- D1 = (top-3 positions, k): 720 x 11 = 7,920 (T274b's rank_exact classes, my arrangement).
- D2 = D1 x (signs of x_8, x_9) where consistent: <= 31,680.
- D3 = D1 x (bin of x_(2) - x_(3) in 2 model-free halves at its conditional median given k): 15,840.
- Witness variant W (the DRI's "weight-derived"): for D1, the gate pattern at the class's conditional-mean input propagated
  with frozen gates. This equals the witness verdict at m = 0, so it is not a separate estimator. Reported as the frozen-gate
  ideal (perfect integration) per design.
- Exact first steps (diagnostic, D1 classes, perfect integration): steps 1..f exact, f+1..10-m frozen at the witness gates, last m
  exact, for f = 1..4 and m = 0, 2, 4.

**Go line.** A frozen E1 design passes iff F <= 10 on all six LB3 groups at its C_eff = max(C, 1k). If any passes, it is frozen
as est(model, budget) -> (estimate, cost_flops) for HP2 scoring. If none passes, the (cost, RMS) curve and the per-group F are
the result, together with the perfect-integration F_ideal = RMS_ideal^2 x C_eff / p(1-p), which says whether better integration
could rescue the design. Repeat on DEV2 when it lands.
