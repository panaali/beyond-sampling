# T403: truth-blind check of the graded-g12 rare-event rule on fresh tie-dominated points

↑ [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) (H-R1, T366, T367, T369, T371, T388) · [Approaches](../README.md)

Owner: Claude child session_01HnQqvN2FmXDdQNcaujmdXV (branch `claude/t403-rare-tie-check`).

## Question

T366's graded-g12 rule (`estimator_t366`: T364's K3 walk estimator with g12 integrated by 10 Gauss-Legendre nodes
clustered on [0, .4] plus a 3-node law rule beyond) is the known fix for the triple-tie sliver (g12 ~ 0) where plain K3
misses by up to 5 orders. T388 found it within 2x on 143/144 noisy points, but on truths the agent already knew; T367
saw it overshoot (2.1-2.4x) where the triple-tie share is high but partial (s547). Is the frozen rule within 2x on
**fresh** (model, threshold) points chosen **before any truth exists**, by the estimator's own predicted tie share?

## Pre-registration (2026-10-19, pushed before any estimate on a panel model or any truth is computed)

**Estimator (frozen, deductive).** `estimator_t366` from claude/t218-tie-tail-labels @5b4a1250 (r1_cont/) with its
pinned dependencies from claude/t286-cheap-frontier @620eb47b, copied unmodified into `frozen/`: the 7 sha256 of
`frozen/r1_cont/t366_freeze_sha256.txt` plus cf.py's (72fb793a...) are asserted at import (`frozen_est.py`).
Settings are the frozen ones (fixed walk, T354 segment start, T_hi = 6). **Disclosed wrapper:** `frozen_est.est_split`
replicates `t366.est_e1`'s loop with rows walked in chunks of 40,000 (as T388's `t388_chunked.py`, for the 2 GB cap),
calling the frozen `walk_seg` and `rare.mass` unmodified, and additionally sums the rows whose graded g12 node is
< 0.1 (7 of the 13 nodes per v2 node: the 6 in panels [0, .07] and the lower node of [.07, .15] at .0869). The total
must equal `estimator_t366.est` to round-off: checked on an excluded model (`python frozen_est.py --check dev3_s526`,
`results/check_dev3_s526.json`) and compared with T388's sealed s526 estimates.
**Predicted triple-tie share** s = (mass from nodes with g12 < 0.1) / total: deterministic, from the estimator alone.

**Models (36).** DEV3 retrains **dev3_s564 ... dev3_s599**: the only DEV3 seeds that appear in no rare-event INDEX row,
README or result file (checked by grepping all rare-event folders, INDEX.md and TASKS.md on the DRI branch and the
T218/T241/T286/T371/T372/T386 branches; they appear only in the DEV3 manifest). **Excluded:** dev3_s500-s547 (used by
T340-T388: T350/T358/T360/T363 E2/E1 panels, T364 s526/s527, T366/T368 s528-s531, T367 screen s532-s547 and selection
s541/s543/s546/s547, T371/T388 s526/s528/s530); dev3_s548-s563 (reserved for T370 stage 2). HP5 (seeds 400-499, the
sealed reserve) is not used. Weights: claude/t286-cheap-frontier rare_events/ext/dev3/weights.npz (sha256 62c95203...,
asserted, `models.py`).

**Event and grid.** E1(t0) = P(pick = position of rank 3 AND g23 > t0), x ~ N(0, I_10). Fixed grid
**t0 = 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5**; one estimator call per model gives all 7 (`est_run.py`). Charged cost per
point: the **whole call's** FLOPs / 5948 (conservative: the other 6 thresholds come almost free).

**Selection (`select_points.py`, deterministic, from the sealed estimates only).**
1. Feasible points: est > 0 and 1000 expected hits need <= 6e8 conditional samples (1000 Z(t0) / est, Z = P(g23 > t0)).
2. Stratum A: 0.5 <= s < 0.8. Stratum B: s >= 0.8.
3. One candidate per model per stratum: A: the t0 with s closest to .65; B: the t0 with the largest s (ties: smaller
   t0). A model can give at most 2 points (one per stratum).
4. If a stratum has more than 10 candidates, keep 10 spread evenly over the s-sorted list (indices
   round(linspace(0, n-1, 10))).
5. Fallback if a stratum has < 6: run the estimator on the extended grid t0 = 2.75, 3.0 on all 36 models
   (`est_run.py S N 2.75,3.0`), and reselect over both grids. If still < 6, score what exists and report the shortfall.

**Seal.** Estimates (est, s, C in FPE) and `results/selection.json` are pushed before any truth is drawn.

**Truth (own code, `truth.py`, sampling-based; not the walk family).** v3-conditional MC: v3 from its exact density
360 phi Phi^7 (1 - Phi(v + t0))^2 / Z(t0) (inverse of a tabulated CDF), two tops iid N | > v3 + t0, seven lows iid
N | < v3, uniform random positions; float64 forward; p = Z hits / n. Common random numbers across models at a t0;
seed 1. Sample size per point: n0 = 2000 Z / est rounded up to 1e7 (min 2e7, max 1e9), i.e. ~2000 hits if the
estimate is right. If a point has < 1000 hits it is extended to the n where the observed rate gives 1000 hits (cap
2e9); if still short, the s.e. is reported. Also reported: truth tie share = hits with g12 < 0.1 / hits.
Code checks (excluded model dev3_s526, before any panel truth): matches T371's independent truth at t0 = 1.0
(1e7 samples: 8.69e-5 +- .06 vs T371's 8.75e-5 from 1e8; tie share .77 vs .77), and plain unconditional MC
(`truth.py --plain 1.0 4e7 7 dev3_s526`, `results/plain_check_*.json`).

**Pre-registered line (fixed now).** Over all selected points: **>= 80% within [1/2, 2] = GO**, else NO GO.
Reported with it (`score.py`, written before any estimate): split by stratum (predicted s) and by truth tie share
(< 50%, 50-80%, >= 80%); every overshoot above 2x; medians, mean log10 ratio, high/low counts; cost (FPE) and the
events plain MC expects at that cost (p C); **F** = MSE ratio vs plain sampling at equal FPE (Directive 8 amendment 2:
F = mean(e^2) C_eff / mean(p(1-p)), C_eff = max(C, 1k), C the mean charged cost; also per point
F_i = e_i^2 C_i / (p_i(1-p_i))), judged against both bars, **strict F <= 1** and **lenient F <= 10**.

**Expectations and alternatives (rule 3), written before running.**
- (a) If T388's result transfers, >= 80% within 2x in both strata, ratios near 1, one-signed high by ~1.1-1.3x (T366).
- (b) Alternative 1 (T367/T369 closure overshoot): in stratum A, near s ~ .7-.8, the rule over-estimates 2-2.5x
  (lows pinned at mu(v4) enlarge triple-tie islands). Signature: A misses high, B fine.
- (c) Alternative 2 (residual quadrature or closure low bias, as T388's orig family, .36-.88): misses low where the
  truth's tie share is partial. Signature: low misses in A.
- (d) Alternative 3 (selection artefact): s from the estimator can disagree with the truth tie share; a point the
  estimator calls tie-dominated may not be. Reported as the s vs truth-tie scatter.
- F expectation: at C ~ 2-4M FPE and p ~ 1e-6 to 1e-8, plain MC expects p C ~ .01-10 events, so F ~ p C (r - 1)^2 is
  likely <= 1 where p C << 1 and the ratio is within 2x; the panel F is dominated by the largest-p points.

## Method

- `frozen/`: unmodified copies of the frozen code (hashes asserted).
- `models.py`: the 36-model panel. `frozen_est.py`: hash checks, `est_split` wrapper, `--check`.
- `est_run.py SHARD NSHARD`: estimates -> `results/est_SHARD.json`.
- `select_points.py`: selection -> `results/selection.json`.
- `truth.py T0 N 1 MODELS`: truths -> `results/truth_t{T0}_seed1.json`.
- `score.py`: -> `results/score.txt`, `results/score.json`.

Environment: `.venv` with numpy and scipy; every run under `ulimit -v 2500000`, at most 2 heavy processes.

## Status

Pre-registered; running.
