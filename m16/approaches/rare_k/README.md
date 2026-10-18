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

## Results
To reproduce: `python3 run.py` (estimates, ~35 CPU-min), then `python3 score.py` (`results/score.json`,
`logs/score.log`). The truths are T192 GPU: t349 @12e28ebb and t358 @f86b0f40.

**Check: K = 1 reproduces my T350 variant B on E1** (relative difference ~1e-5, from the Stieltjes grid), on seed0.

**Calibration on T340's 3 models** (my K3, against T340c's pre-fix K3 numbers):
- orig E1: 1.77 / 1.24 / .48, against T340c's 1.73 / 1.24 / .49.
- orig E2: 1.03 / 1.01 (after the walk fix), against T340c's pre-fix 1.30 / 1.61.
- seed0 E1: 1.32 / 1.44 / 1.43; E2: .97 / .98.

These agree with T340c on E1. My E2 matches T340c's post-fix report (E2 6/6 within 2x).

**Verdict: GO at K = 3, exactly on the line.**
- 88/110 = **80%** within 2x.
- Medians: all .76, E1 1.03, E2 .73. All lie in [2/3, 1.5], but E2's is close to the lower edge.
- K = 5 is the same: 80%, medians .76 / 1.09 / .72.

| event | n | within 2x (K3) | median est/truth (K3) | median (K5) |
|---|---|---|---|---|
| E1, t0 = 1 | 6 | 6 | .99 | 1.09 |
| E1, t0 = 1.25 | 6 | 6 | 1.05 | 1.11 |
| E1, t0 = 1.5 | 6 | 6 | 1.06 | 1.14 |
| E2, t = .5 | 26 | 25 | .74 | .73 |
| E2, t = .75 | 26 | 19 | .69 | .70 |
| E2, t = 1.0 | 20 | 13 | .70 | .71 |
| E2, t = 1.25 | 20 | 13 | .69 | .70 |

- **E1, the far-gap islands, is clean: 18/18 within 2x, unbiased (medians .99–1.06).** This fixes T350's E1 failure,
  where variant B on dev3_s505 was 6–10x low. g12 integration is what closes it, as T340c found.
- **E2, the interior error set, is one-signed low.**
  - 79% of E2 cases are below 1; 70/92 are within 2x.
  - The 22 misses sit in 9 models. On 7 of them the ratio collapses as t rises (dev3_s515: .39 / .20 / .07 / .01;
    s512 and s518 fall to .07 and .03 at t = 1.25). Two models over-estimate (s508 3.0 / 4.3; s519 2.6).
  - This is the same pattern as variant A in T358, at roughly 250x A's cost. g12 integration does not reach the
    deeper interior islands on these models.
  - Alternative 2's predicted overshoot did not appear. The sign is the opposite.
- **K = 5 does not change anything**, so this is not quadrature error in g12. The residual lies elsewhere: the
  single v3* per v2 node, the v3–v4 link, or the lows.

**Secondary: the equal-compute comparison** (T350's rule, at executed C):
- K3 wins 21, ties 22, loses 67 (share .24). K5 wins 16, ties 18, loses 76.
- Executed C is 1.0–1.5M FPE for the three E1 events together, and 3.7–6.7M FPE for the four E2 thresholds together.
- Each case is charged its family's total, which is conservative. Charging a quarter of C_E2 per threshold would
  not change the sign: at these costs sampling sees tens to thousands of events.
- **The accurate estimator is not competitive with sampling at equal compute, except at the rarest p.**

## Claims
- [measured] On 26 fresh DEV3 models, my implementation of T340c (K = 3) is within 2x on 88/110 (model, event)
  cases. It passes the DRI's go line exactly (≥ 80%; medians .76 / 1.03 / .73 within [2/3, 1.5]).
- [measured] E1 (far-gap rank-3 islands, p ~ 1e-6 to 2e-4): 18/18 within 2x, median about 1.
- [measured] E2 (interior errors with both gaps ≥ t): 70/92 within 2x, one-signed low (median .69–.74). On 7 models
  it collapses toward 0 as t rises. K = 5 does not fix it.
- [measured] At 1–7M FPE it loses to sampling at equal compute in most cases (21 win / 22 tie / 67 loss).

## Alternatives not ruled out
- The E2 deficit may come from the node-specific v3* (one g23 transverse point per v2 node) and not from g12.
  Integrating g23 (v3) at Gauss nodes too was not tried.
- Six models for E1 is a small sample: 18/18 has a 95% lower bound of about 85% within 2x.

## Compute accounting
`run.py`: 29 models × (K3 + K5), about 35 CPU-min in total, single thread, `ulimit -v 2500000`. Estimator costs are
as charged above.

## Status
**Done. GO at K = 3, exactly on the 80% line.** E1 is clean. E2 is one-signed low and fails on about a quarter of
models at t ≥ .75. The method is not competitive at equal compute. Unverified.
