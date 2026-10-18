# T286: cheap-regime frontier of the existing deductive / mechanistic estimators

↑ [Approaches index](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md)

## Question
Scored by its own cost (TASKS.md Directive 8 amendment 2), a fixed deductive estimator passes iff
F = mean(e^2) * C_eff / mean(p(1-p)) <= 1 (strict) or <= 10 (lenient) on a group of models, with
C = charged FLOPs / 5948 (FPE, T196) and C_eff = max(C, 1000). Every existing mechanistic estimator was tuned for
16k-256k FPE. What are their (cost, error) points at their *cheapest* settings, which error components dominate, and
is any of them close to F <= 1 / F <= 10?

## Pre-registration (written 2026-10-18 before any sweep was run)
Already looked at before writing this (disclosed): the LB3 raw runs (`results/leaderboard_v3_raw.jsonl`) for
rank_exact, observer, reliability, cone_enum, propagation, branch_bound, arc_cumulants; the T274b `cone_bias.log`
and the probe_v5 README tables. Nothing was run.

**Panel.** LB3 test models: orig, seed0-4, s{.0025,.005,.01,.02}_{0..5} (30 models, `m16/leaderboard.py`),
truths `results/leaderboard_truths.json` (32M samples). Then DEV2 (T284) when published. HP1/HP2 truths never read.

**Score.** Per group g and setting: bias_g, RMS_g (truth s.e. subtracted as in LB3), C_g = mean charged FPE,
F_g = RMS_g^2 * max(C_g, 1000) / mean_g p(1-p); 95% CI by bootstrap over the group's models.
F_max = max over the six groups.

**Frontier (fixed now).** (a) The Pareto set of all (estimator, setting) points in (C, F_max); (b) per group, the
Pareto set in (C, RMS_g). The **candidate** is the setting with the lowest F_max among settings whose steps are all
labelled deductive (exact computation or a declared deductive approximation, no averaging of model outputs over a
fixed node set). The best setting that includes an inductive step is reported separately and is not the candidate.
Ties within 10% of F_max are broken by lower C.

**Settings grid (fixed now; cheapest first).**
- rank_exact: Q in {4, 8, 16, 64}; position-triple subset stride s in {1, 2, 4, 8, 16} (balanced: triples j with
  j mod s = 0 in a fixed position-balanced order); K-classes per-K (11) or pooled into 3 groups or 1 group.
- observer v4: q in {1, 2}; nq in {6, 8, 12}; relative-gap bins 6 (default) or 3; triple stride {1, 3, 8, 24}.
- reliability: LB3 ladder L0 and cheaper L0 variants (chamber stride x2/x4/x8, modes (0,2), SORM off).
- cone_enum, propagation, branch_bound: their LB3 cheapest-budget rows (cost already at or near the floor).
- T274b deductive root-cone baseline (180 cones, m = 0..2 exact last steps): its measured bias with the
  implemented-cost estimate.
- probe series: `m16/estimate.py` v1-v4 (bisection probes) with iters in {4, 8, 12} and triple strides; probe_v5b
  (deductive spread, fixed nodes) and v5a / v5q (inductive references) from their LB3 raw.

**Cost.** Charged as implemented: every operation that depends on the weights (forward passes, tangents, bisection
probes, Jacobians, cone intervals, quadrature of weight-derived maps) counted in FLOPs (multiply-add = 2), checked
against CPU time relative to plain sampling.

**Expected (before measuring).** No existing estimator reaches F_max <= 1. rank_exact at ~1-3k FPE: frozen-gate bias
~1-2 pp, F_max ~5-20. observer at its cheapest: ~5-10k FPE with ~1 pp, F ~10-30. cone_enum, propagation,
reliability and the cone baseline: F_max >> 10, dominated by the seed group.
Alternatives to keep in view: (1) a small cheap-setting RMS may be cancellation between components, not accuracy;
(2) triple subsampling is fixed-node averaging over positions (inductive); its error should be separated from the
structural bias; (3) settings are picked on test models (LB3), so only HP2 is a clean score.

## Method
(filled in after measurement)

## Status
Running.
