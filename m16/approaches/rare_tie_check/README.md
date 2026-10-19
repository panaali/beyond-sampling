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

## Seal: selection and amendment 1 (pushed before any truth is drawn)

- Estimates: `results/est_{0,1}.json` (36 models x 7 t0) and, by the pre-registered fallback (stratum B had 5 < 6),
  `results/est_ext_{0,1}.json` (t0 2.75, 3.0). Selection: `python select_points.py` -> `results/selection.json`.
- **Stratum B stays at 5 after the fallback**: only 6 of 36 models have any point with s >= .8, and dev3_s595's are
  infeasible (est ~1e-12 to 1e-15). By the pre-registration the primary line is scored on what exists: **15 points
  (A 10, B 5)**; the B shortfall (5 vs the target 6) is reported.
- **Amendment 1 (secondary, written after seeing only the estimates, before any truth).** B2: for each model with
  another feasible B point, the unselected one with the smallest t0 (dev3_s570 @1.0, dev3_s585 @1.0, dev3_s593 @1.25).
  Scored separately (`score.py`, 'secondary'); it does not enter the pre-registered line.

| stratum | model | t0 | est | s (pred. tie share) | C (M FPE) |
|---|---|---|---|---|---|
| A 50-80% | dev3_s576 | 1 | 1.379e-04 | 0.505 | 2.55 |
| A 50-80% | dev3_s564 | 2.75 | 3.067e-08 | 0.511 | 1.39 |
| A 50-80% | dev3_s580 | 1.25 | 9.720e-05 | 0.521 | 3.12 |
| A 50-80% | dev3_s583 | 1.5 | 4.041e-06 | 0.624 | 2.32 |
| A 50-80% | dev3_s596 | 2.5 | 4.690e-08 | 0.652 | 3.06 |
| A 50-80% | dev3_s565 | 2 | 8.980e-07 | 0.658 | 2.68 |
| A 50-80% | dev3_s571 | 1 | 6.167e-06 | 0.663 | 2.49 |
| A 50-80% | dev3_s595 | 1.75 | 2.980e-08 | 0.675 | 3.25 |
| A 50-80% | dev3_s590 | 1.5 | 2.272e-06 | 0.688 | 2.93 |
| A 50-80% | dev3_s593 | 2.75 | 8.990e-09 | 0.708 | 1.65 |
| B >=80% | dev3_s596 | 1 | 4.835e-05 | 0.829 | 3.06 |
| B >=80% | dev3_s583 | 1 | 3.343e-05 | 0.830 | 2.32 |
| B >=80% | dev3_s593 | 1 | 2.390e-05 | 0.846 | 2.60 |
| B >=80% | dev3_s585 | 1.25 | 8.594e-08 | 0.971 | 2.95 |
| B >=80% | dev3_s570 | 3 | 1.048e-10 | 1.000 | 1.63 |
| B2 >=80% (secondary) | dev3_s570 | 1 | 2.783e-06 | 0.891 | 2.59 |
| B2 >=80% (secondary) | dev3_s585 | 1 | 6.706e-07 | 0.942 | 2.95 |
| B2 >=80% (secondary) | dev3_s593 | 1.25 | 9.616e-06 | 0.837 | 2.60 |

## Method

- `frozen/`: unmodified copies of the frozen code (hashes asserted).
- `models.py`: the 36-model panel. `frozen_est.py`: hash checks, `est_split` wrapper, `--check`.
- `est_run.py SHARD NSHARD`: estimates -> `results/est_SHARD.json`.
- `select_points.py`: selection -> `results/selection.json`.
- `truth.py T0 N 1 MODELS`: truths -> `results/truth_t{T0}_seed1.json`.
- `score.py`: -> `results/score.txt`, `results/score.json`.

Environment: `.venv` with numpy and scipy; every run under `ulimit -v 2500000`, at most 2 heavy processes.

## Status

**Done (2026-10-19). GO by the pre-registered line, exactly on it (12/15 = 80%), unverified.** Pre-registered
71cf5547; estimates and selection sealed 62198f17 before any truth; truths and scoring after.

## Results (`python score.py` -> `results/score.txt`, `results/score.json`; tie split `results/tie_split.txt`)

| set | n | within 2x | median | mean log10 | > 2x | < .5x |
|---|---|---|---|---|---|---|
| **all primary (line: >= 80%)** | 15 | **12 (80%)** | .95 | -.08 | 1 | 2 |
| predicted share 50-80% (A) | 10 | 9 (90%) | .98 | +.04 | 1 | 0 |
| predicted share >= 80% (B) | 5 | 3 (60%) | .86 | -.32 | 0 | 2 |
| truth tie share < 50% | 7 | 5 | 1.01 | -.09 | 1 | 1 |
| truth tie share 50-80% | 6 | 6 | .95 | -.02 | 0 | 0 |
| truth tie share >= 80% | 2 | 1 | .64 | -.23 | 0 | 1 |
| amendment 1, B2 (secondary) | 3 | 2 | .63 | -.38 | 0 | 1 |
| primary + B2 (secondary) | 18 | 14 (78%) | .92 | -.13 | 1 | 3 |

Per point (est / truth; s = predicted tie share, tie = truth tie share; C in M FPE; every truth >= 1,011 hits,
s.e. <= 3.1%):

| point | ratio | s | tie | truth | C | p C | F_i |
|---|---|---|---|---|---|---|---|
| A s576 @1.0 | 1.20 | .51 | .47 | 1.15e-4 | 2.55 | 290 | 11.7 |
| A s564 @2.75 | 1.03 | .51 | .46 | 2.98e-8 | 1.39 | .04 | 3e-5 |
| A s580 @1.25 | 1.01 | .52 | .46 | 9.59e-5 | 3.12 | 300 | .06 |
| A s583 @1.5 | .98 | .62 | .59 | 4.11e-6 | 2.32 | 9.5 | .003 |
| A s596 @2.5 | .88 | .65 | .60 | 5.36e-8 | 3.06 | .16 | .003 |
| A s565 @2.0 | .97 | .66 | .59 | 9.27e-7 | 2.68 | 2.5 | .003 |
| A s571 @1.0 | .95 | .66 | .37 | 6.50e-6 | 2.49 | 16 | .04 |
| **A s595 @1.75** | **5.04** | .67 | .37 | 5.91e-9 | 3.25 | .02 | .31 |
| A s590 @1.5 | .56 | .69 | .47 | 4.04e-6 | 2.93 | 12 | 2.3 |
| A s593 @2.75 | .94 | .71 | .71 | 9.59e-9 | 1.65 | .02 | 6e-5 |
| B s596 @1.0 | .86 | .83 | .65 | 5.63e-5 | 3.06 | 170 | 3.5 |
| B s583 @1.0 | 1.11 | .83 | .74 | 3.02e-5 | 2.32 | 70 | .81 |
| B s593 @1.0 | .90 | .85 | .85 | 2.67e-5 | 2.60 | 70 | .76 |
| **B s585 @1.25** | **.08** | .97 | .30 | 1.14e-6 | 2.95 | 3.4 | 2.9 |
| **B s570 @3.0** | **.39** | 1.00 | .97 | 2.71e-10 | 1.63 | 4e-4 | 2e-4 |
| B2 s570 @1.0 | .63 | .89 | .83 | 4.45e-6 | 2.59 | 12 | 1.6 |
| **B2 s585 @1.0** | **.14** | .94 | .31 | 4.85e-6 | 2.95 | 14 | 10.6 |
| B2 s593 @1.25 | .86 | .84 | .83 | 1.13e-5 | 2.60 | 29 | .62 |

- **Misses (3 primary).** (1) dev3_s595 @1.75, **overshoot 5.0x** (s .67, truth tie .37): the tie part is ~9x high
  (`tie_split.txt`), T367's s547 pattern (T369: closure enlarges the triple-tie islands), here larger. (2) dev3_s585
  @1.25, **.08** (and @1.0 .14, secondary): the estimator puts 97% of its mass in the sliver, but 70% of the true
  hits have g12 >= .1 and the estimator has essentially **no** mass there (non-tie part ~.00-.01x), and its tie part is
  also low (.25-.41). (3) dev3_s570 @3.0, **.39** (pure sliver, truth tie .97; p 2.7e-10): one-signed low in the deep
  tie limit, as T366's deepest target point (.24).
- **Predicted vs truth tie share:** s tracks the truth's within ~.1 on 12/18 points, but over-predicts on s585
  (.94-.97 vs .30), s571 (.66 vs .37), s595 (.67 vs .37) and s596 @1.0 (.83 vs .65). The selection variable is not
  reliable where the estimator is wrong; 2 of 5 B points are not truth-tie-dominated.
- **Bias sign:** no one-signed bias overall (median .95; 6 high / 9 low), unlike T366's 1.1-1.3x high.
- **F (plain MC at equal FPE, Directive 8 amendment 2):** panel **F = 4.57: strict bar (<= 1) FAIL, lenient bar
  (<= 10) PASS**. The panel figure is dominated by the largest-p point (s576 @1.0: 1.20x at p C = 290, F_i 11.7).
  Per point: F_i <= 1 on 11/15, <= 10 on 14/15. Where p C << 1 (6 points, p <~ 5e-8) the estimate beats sampling by
  orders of magnitude whenever it is within ~2x (F_i <= .31, including the 5x overshoot since sampling would see ~0 events).
- **Cost:** 1.39-3.25M FPE per charged point (the whole call, all 7 or 2 thresholds); the selection scan was 36 x 7
  + 36 x 2 (fallback) estimator calls, 165M FPE, 6.0 CPU-h. Truths ~2.4e9 conditional model forwards, ~2.5 CPU-h.

**Reproduce.** `python est_run.py 0 2; python est_run.py 1 2` (and `... 2.75,3.0` for the fallback);
`python select_points.py`; truths: `python truth.py T0 N 1 MODELS` with the (t0, n, models) of `results/truth_*.log`
(1.25: 4e8; 1.0: 1e8; 1.75: 1e8 then extended to 3.3e8 by the < 1,000-hit rule; 3.0: 4e7; others 2e7); `python score.py`.

## Compute accounting

Estimator (deductive, no random model inputs): FLOPs as executed by the frozen code / 5948 (FPE), 1.26-3.51M FPE per
call; model-free tables (cont_g TABLES, graded node lines) uncharged as in T366. Truth (sampling): v3-conditional MC,
float64 forward, ~2.4e9 model forwards. Checks: wrapper vs `estimator_t366.est` on excluded dev3_s526 (equal to 1e-16,
C equal to T388's), truth code vs T371's truth and vs plain MC on dev3_s526 at t0 1.0 (8.69e-5 +- .06 vs 8.75e-5 and
8.66e-5 +- .15). Total CPU ~9 h, at most 2 heavy processes, each under `ulimit -v 2500000`.

## Claims

1. [measured] On 15 truth-blind points (36 fresh DEV3 models, selected by the frozen estimator's own predicted tie
   share), the frozen graded-g12 rule is within 2x on 12 (80%): GO, exactly on the pre-registered line.
2. [measured] It is good at partial predicted tie share (A: 9/10, median .98) and weaker at high predicted share
   (B: 3/5, misses .08 and .39); one overshoot (5.0x, s595, partial tie share) reproduces the s547 pattern at a
   larger size.
3. [measured] Panel F = 4.57: fails the strict bar, passes the lenient bar; 11/15 points F_i <= 1.
4. [inferred] Three distinct failure modes remain: closure overshoot on partial tie (s595), a model whose non-tie
   islands the walk family misses entirely (s585, its own large error, with an over-predicted tie share), and the
   deep pure-sliver low bias (s570 @3.0). The tie / non-tie split compares quadrature nodes g12 < .1 with hits
   g12 < .1, an approximate correspondence.
5. [measured] The estimator's predicted tie share is not a reliable proxy for the truth's (s585 .97 vs .30).

## Alternatives not ruled out

- The GO is on the line: one more miss would be NO GO; the 95% interval for the within-2x rate at 12/15 is ~.52-.96.
  With the secondary B2 points it is 14/18 (78%), below the line.
- s585's miss could be the walk geometry (lines placed at the closure's transverse point miss a non-tie island), not
  the g12 rule; an oracle switch-off (T363/T369 style) on s585 would separate them. Not done.
- s595's overshoot could be closure (T369's mechanism) or a v4-position effect; not separated.
- Truth code shares the conditional-law construction with T371 (written independently); it was checked against plain
  MC on one model and one threshold only.
- B is short (5 points from 5 models); only 6 of 36 models have any point with s >= .8, so tie-dominated fresh points
  are scarce at feasible p.

## Failure modes

- Points where the estimator's own tie share is wrong (s585): it cannot flag its own failure.
- Partial tie share with a closure-enlarged sliver: over-estimates (up to 5x here).
- Deep pure-sliver points (p ~1e-10): ~.4x low.
- Cost 1.4-3.3M FPE per call: at p >~ 1e-5 plain MC at the same cost sees hundreds of events and wins on F.
