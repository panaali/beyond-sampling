# T349: independent check of T340's rare-event result on fresh DEV3 models

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · T340 (claude/t286-cheap-frontier 620eb47b, rare_events/)

## Question
T340 found that T317's exact walks, uncut, estimate rare error probabilities (1e-4 to 1e-7) better than sampling at
equal compute in about 12 of 15 (model, event) cases on 3 models:
- variant B for the far-gap rank-3 islands E1;
- variant A for the interior error set E2.

Does this hold on ≥ 6 fresh models, with an independent implementation? I use my own walk (`r1w_repro/r1w_indep.py`,
T312) and have not read T340's code (`rare.py` etc.), only its README.

## Pre-registration (2026-10-18, before any DEV3 model or truth exists)
**Events** (as T340 defines them; v1 > v2 > v3 the top 3, g12 = v1 − v2, g23 = v2 − v3):
- **E1(t0)** = {the network outputs the position of the 3rd-largest input} ∩ {g23 > t0}, for t0 ∈ {1, 1.25, 1.5}.
- **E2(t)** = {the network is wrong (any pick)} ∩ {g12 ≥ t} ∩ {g23 ≥ t}, for t ∈ {.5, .75}.

**Estimators**, my derivation from T340's README. Both are deductive, with cost charged as executed.
- Common to both:
  - The line-23 inputs follow T317's closure: v1 = v2* + g12*, v2*, v3 = v2* − t, v4 = v3 − δ at the latest free
    position, and lows at μ(v4*) + μ'(v4*)(v4 − v4*).
  - My exact segment walk (T312) runs with **no tmax cut**, to t = 6, with no screen.
  - Each cell's affine logits give the exact interval where a given position is the argmax.
  - The masses use the exact model-free law: P = Σ_triples ∫ dv2 φ(v2) · M12(v2) · Σ_intervals
    [Φ(v2 − a)^8 − Φ(v2 − b)^8]/8. Both regions (nb: v2 ∈ (0, 6.5], b: [−6.5, 0]) use 20 Gauss-Legendre nodes each.
- **E1 by variant B.** One walk per triple per GL node, with node-specific transverse points:
  - g12 = E[v1 − v2 | v2], in closed form;
  - v3* = v2 − E[v2 − v3 | v2], by quadrature;
  - v4* = E[max of 7 below v3*], with δ = v3* − v4*.

  Intervals are where the argmax is p3, restricted to t > t0. M12 = 1 − Φ(v2), the full g12 mass.
- **E2 by variant A.** One walk per triple per region, at T317's region transverse point (the T317e TABLES constants).
  Intervals are the complement in [t, 6] of the set where p2 is the argmax. M12 = 1 − Φ(v2 + t) (g12 ≥ t;
  independence declared).
- **Cost (FPE)** is charged as executed:
  - walk cells as in T312: 2 FPE for the first cell, 2(T − s)/T for a restart at step s;
  - plus 20 FLOPs/5,948 per Φ evaluation at a weight-dependent point (interval endpoints × G nodes).
  - Model-free tables are uncharged.

**Metric**, per (model, event) case, with truth p and my executed cost C:
- relative error r = |est/p − 1|;
- the competitor's relative error s = min(1, 1/sqrt(pC)). Sampling at C forward passes has relative sd 1/sqrt(pC);
  when pC < 1, the zero answer (relative error 1) is the better competitor.
- **The case is won if r < s.**
- The truth's own MC error is propagated as a sensitivity: the case is "won robustly" if it is won at both ends of
  the truth's ±2 SE.

**Models.** The **first 6 models in DEV3's manifest order** (T342, built by the T192 child), whatever their names.
**Truths:** GPU MC with ≥ 1e9 inputs per model, all five events counted at once (requested from T192).

**Go:** won in ≥ 2/3 of the 30 cases (≥ 20). Also reported: per event family (E1: 18 cases, E2: 12), and the
ratio est/truth against T340's bands (useful within 2x, partial within 3x).

**Calibration before DEV3 (not part of the verdict):** I run the same code on T340's three models (orig, seed0,
seed3; T192's weights and GPU truths at 63d36d38, t340/) and compare my est/truth ratios with T340's table. Agreement
within about 20% would show that the two implementations match. A disagreement is reported, and the DEV3 run uses my
code as pre-registered either way.

**Alternatives first.**
1. T340's wins come from three models with a favourable transverse placement. On fresh models the far-gap islands
   (E1) may sit elsewhere, and variant B loses more often than on the seeds.
2. The metric favours estimators that are cheap relative to 1/p. If my implementation is costlier than T340's, it
   wins less often for the same accuracy. C is reported per case so that this can be separated.

## Status
Pre-registered; calibration next.
