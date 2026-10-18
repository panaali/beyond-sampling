# T360: independent test of the K-node rare-event estimator (T340c) on fresh DEV3 models, under the within-2x criterion

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [T350](../rare_check/README.md) · [T358](../rare_e2/README.md)

## Question
T340c (T349 by the T218 child, rerun after the walk fix) replaces variant B's pinned g12 = E[g12 | v2] by a K-node Gauss
rule for the exact law of g12 | v2. On T340's 3 open models it reports K3 within 2x on 8/9 E1 and 6/6 E2 pairs.

Does that hold on fresh models? The test uses my own implementation, with no r1_cont or T340 code copied, and the
DRI's new primary criterion: est/truth within 2x.

## Pre-registration (2026-10-18, before any T360 estimate is computed)
**Disclosure.**
- I have already seen the GPU truths for all the models used here, through T350 (dev3_s500..505) and T358
  (dev3_s506..525).
- I have also seen T340c's README design and its results on T340's 3 models.
- The estimator is fully specified by T340c's README (r1_cont/README "T349 (T340c) pre-registration"). The only free
  choice, K, is fixed here (K = 3 primary, K = 5 check), so there is no tuning on these truths.

**Estimator** (my implementation, on my T312 walk, `../rare_check/rare.py` helpers; deductive; cost as executed):
- Variant B as in T350:
  - per v2 Gauss-Legendre node (20 per region, nb and b), the node-specific v3* = v2 − E[g23 | v2];
  - v4* = E[max of 7 below v3*];
  - the line-23 closure with the v3–v4 link and linearised lows;
  - uncut walks to t = 6.
- **Change (T340c):** g12 is integrated with a **K-node Gauss rule for the exact law of g12 | v2**, density
  ∝ φ(v2 + g) on g ≥ 0 (Golub–Welsch on a fine model-free grid). There is one line-23 walk per (triple, v2 node,
  g12 node), with weight ω_k.
- E1(t0): argmax = p3 intervals restricted to t > t0. Mass factor (1 − Φ(v2)) Σ_k ω_k.
- E2(t): the rule is for g12 | v2 restricted to g12 ≥ t, with total mass 1 − Φ(v2 + t), one set of walks per t. The
  wrong set is the complement of argmax = p2 within [t, 6].
- **K = 3 primary; K = 5 reported as a convergence check.**

**Cases and truths** (T192 GPU MC, 1e9 float64 inputs per model):
- E1(t0 = 1, 1.25, 1.5) on dev3_s500..505: 18 cases (t349/rare_mc.json @12e28ebb).
- E2(t = .5, .75) on dev3_s500..505: 12 cases. T350's truths have no E2 at t = 1 and 1.25 for these models, so those
  cases are dropped.
- E2(t = .5, .75, 1, 1.25) on dev3_s506..525: 80 cases (t358/rare_mc.json @f86b0f40).
- Total: **110 cases**.

**Go (DRI):**
- (1) ≥ 80% of the 110 cases have est/truth within [1/2, 2];
- (2) no systematic one-signed bias: the median est/truth over all cases, and within each event family (E1, E2),
  lies in [2/3, 1.5].

**Secondary.** The equal-compute comparison at executed C (T350's scoring rule), and est/truth by threshold.

**Alternatives first.**
1. T340c's accuracy on T340's models came partly from those three models. On fresh retrains the far-gap islands may
   need more than g12 integration (the v3–v4 link, the lows), and E1 falls below 80% within 2x.
2. T340c reports a residual one-signed E2 overshoot rising with t (+12–65%). At t = 1–1.25 on 20 models it may
   exceed 1.5x in the median.

## Status
Pre-registered.
