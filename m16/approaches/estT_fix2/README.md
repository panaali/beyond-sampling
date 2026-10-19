# T443: re-check results that used the old estT d = 4 orthant rule

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [estT_fix (T131/E46)](../estT_fix/README.md)

## Question

T428 found that `understand_m43/estT.py` (the original estT) has quadrature errors of ~1e-4 at T = 4 in its d = 4
orthant masses (12-node 'wide' rule). That is the defect T131 had already fixed in [estT_fix/estT.py](../estT_fix/estT.py)
(log substitution, 128 nodes, cofactor angles; E46). Which results still rest on the OLD rule, and does correcting
them change any claim?

## Method

- **Reference engine:** `estT_fix/estT.py`, unchanged (DRI re-scope 2026-10-19: no new estT copy). The old behaviour
  remains reproducible in `understand_m43/estT.py`. Check: on zoo (4,4) seeds 0 and 2 estT_fix exact equals T428's
  converged 128-node values to 4e-9 (0.4770131, 0.6017307) [measured].
- **T192's GPU port** (branch claude/t192-gpu-cost-law, `gpu_cost_law/estT_base.py` + `estG.py`): built on the
  FIXED orthant (`_LOG_S`, 128 nodes); `_rule(12, 2)` is kept there only as an unused table [measured, by reading].
- **Inventory** (read-only sweep of INDEX, all approach/review folders and all 52 remote branches; old copy =
  d = 4 orthant integrating with `_RULES` (12,2)/(48,3) and no `_LOG_S`): old copies are `understand_m43/estT.py`
  and identical snapshots (`certificate_review/estT_frozen.py`, `final3_review/snapshot/estT.py`,
  `leaf_mass_review/estT_snapshot.py`, `review/split_attribution_review/revised_cap/frozen/.../estT.py`) and the
  older `class_region_families/estT_snapshot.py`. All t192/t194/t195 ports were fixed from their first commit.

## Pre-registered scan list (2026-10-19, before any re-scoring)

| # | result | what used the old rule | action | counts as changed if |
|---|---|---|---|---|
| 1 | **E41 / T87** rung (4,4), zoo seeds 0-4 + 8 noisy each | clean truth (exact) and recipe estimates (clean and noisy) | `rescore.py e41`: exact + recipe with estT_fix; recipe on the SAME noisy copies (noise seed 50) vs the saved 1e7-MC truths | any seed's clean or noisy ratio crosses 1, or the rung verdict changes |
| 2 | **E41 checklist F1, F2** (seed 0) | truth and estimates | `rescore.py f12` | F1 or F2 ratio crosses 1 |
| 3 | **T111 / T428** (15 retrained (4,4) models) | as 1 | `rescore.py t111` | as 1; also T428's 128-node re-score vs estT_fix |
| 4 | **E45 (8,4) part, T133 audit, T108 diagnostics** (`final3/`, `final3_review/`, `understand_m43/diag108*`) | saved old-rule (8,4) truths (zoo0 old exact .9750473 vs fixed .9761608: -1.1e-3) | **note only**: already superseded by E46's fixed re-score; T133's "nine H8T4 failures survive" is moot | n/a |
| 5 | **T110-T4** (`class_region_t4/`, `class_region_families/estT_snapshot.py`) | recipe-mode masses for 3 partitions | **note only**: its runner pins paths/checkpoint of a Codex workspace not present here; its exact cases were replayed with doubled/quadrupled masses by its authors; its recipe-vs-partition difference (2.39e-4) is the same order as the old-rule error, so the ordering is not attributable | n/a |
| 6 | **E47 (T165)** recipe comparator column (reuses run87.json) | old recipe estimates (ratios 18-161) | **note only**: a ~1e-4 shift cannot matter at these ratios; superseded by row 1's numbers | n/a |
| 7 | T131 Codex lineage (T131/T136/T137/T140/T142/T145/T146/T148/T154/T162/T171) | the old rule is their object of study, truths independent | none | n/a |

Not affected (fixed copies or no estT masses): E46, E48, T192, T194 and its T216/T217/T222/T232/T239 scripts, T195,
T196, T46 (own orthant), T373 (geometry only), chamber_split, all T <= 3 results, every result whose (4,4) truth is
the independent .4770131 or Hopf/MC. Latent trap (no numbers affected): `review/split_attribution_review/revised_cap/frozen`
would resolve `import estT` to the old copy if re-run at T = 4.

**Corrected value for INDEX** = full fix (estT_fix truth, estT_fix recipe estimate and its op count). Also reported:
"truth-only" (old recipe estimate and ops, fixed truth), which isolates the truth correction.

## Status

Done (2026-10-19).

## Results

### 1-3. Re-scored rows [measured]

Commands: `python rescore.py e41`, `python rescore.py t111`, `python rescore.py f12` (pre-registered, variant **A** =
estT_fix for truth, recipe estimate and op count) and the same with a trailing `r24` (post-hoc variant **B**, see
below). Exact truth is estT_fix in both. Noisy truths are the saved 1e7-MC values of the original runs.

| model | exact old | exact fix | recipe err (A) pp | clean ratio old | clean A | clean B | noisy ratio old | noisy A | noisy B |
|---|---|---|---|---|---|---|---|---|---|
| E41 zoo_0 | 0.476910 | 0.477013 | -1.414 | 459.04 | **2055.27** | 486.13 | 90.79 | **418.21** | 96.71 |
| E41 zoo_1 | 0.602038 | 0.602018 | -0.619 | 71.13 | **345.31** | 77.84 | 32.92 | **162.68** | 36.24 |
| E41 zoo_2 | 0.601771 | 0.601731 | -0.659 | 75.78 | **369.99** | 83.03 | 14.41 | **70.62** | 15.80 |
| E41 zoo_3 | 0.498079 | 0.497956 | -0.204 | 7.89 | **36.19** | 8.17 | 135.72 | **650.42** | 147.60 |
| E41 zoo_4 | 0.602001 | 0.602000 | -0.668 | 76.20 | **375.88** | 84.15 | 17.33 | **87.67** | 19.55 |
| T111 30sL2_0 | 0.486617 | 0.486582 | -0.029 | 0.18 | **0.80** | 0.18 | 0.91 | **4.25** | 0.96 |
| T111 30sL2_1 | 0.600414 | 0.600359 | -0.710 | 83.26 | **407.97** | 91.00 | 13.42 | **66.84** | 14.95 |
| T111 30sL2_2 | 0.602868 | 0.602826 | -0.750 | 93.14 | **455.26** | 101.96 | 13.08 | **65.23** | 14.62 |
| T111 30sL2_3 | 0.496452 | 0.496334 | -0.226 | 9.62 | **44.77** | 10.09 | 130.89 | **633.71** | 143.81 |
| T111 30sL2_4 | 0.602049 | 0.601996 | -0.517 | 45.61 | **220.06** | 49.68 | 13.81 | **70.06** | 15.62 |
| T111 31s_0 | 0.477622 | 0.477755 | +0.069 | 1.14 | **4.98** | 1.18 | 32.95 | **152.97** | 35.32 |
| T111 31s_1 | 0.599635 | 0.599600 | -0.559 | 49.21 | **243.25** | 54.50 | 18.15 | **89.42** | 20.10 |
| T111 31s_2 | 0.600127 | 0.600097 | -0.582 | 53.21 | **263.26** | 58.93 | 16.64 | **83.03** | 18.62 |
| T111 31s_3 | 0.498803 | 0.498665 | -0.195 | 7.28 | **33.20** | 7.50 | 69.25 | **336.58** | 76.11 |
| T111 31s_4 | 0.600610 | 0.600606 | -0.599 | 55.60 | **277.38** | 62.03 | 15.42 | **76.67** | 17.16 |
| T111 30s_42 | 0.601674 | 0.601646 | -0.694 | 81.62 | **403.33** | 90.19 | 7.65 | **38.68** | 8.58 |
| T111 30s_20 | 0.594403 | 0.594361 | +0.389 | 22.65 | **106.22** | 24.57 | 30.05 | **140.70** | 32.53 |
| T111 30s_40 | 0.561111 | 0.561226 | +0.058 | 0.56 | **1.87** | 0.43 | 1.06 | **5.14** | 1.17 |
| T111 30s_10 | 0.557509 | 0.557666 | +0.093 | 1.10 | **4.08** | 0.94 | 15.19 | **71.84** | 16.46 |
| T111 30s_44 | 0.556965 | 0.557142 | +0.107 | 1.94 | **8.04** | 1.84 | 14.94 | **70.96** | 16.24 |

E41 checklist (seed 0): F1 (eps 0.01) old err -0.105 pp, ratio ~3 -> A -0.108 pp, **15.2** (310M ops) / B 3.61 (74M ops);
F2 (intermediate exact) old -0.106 pp, ~3 -> A **15.4** / B 3.67. Both fail in every variant.

**Why A and B differ (post-hoc, `rulecost.py`):** the recipe's estimates and errors are the same in A and B to ~1e-6;
only the op count differs. estT_fix's d = 4 mass (log substitution, 128 nodes) costs ~4.5x more ops per model than
the old rule. But the old rule's defect is its **p = 2 grading**, not its node count: with p = 3 grading the old engine
is within 5e-6 (12 nodes) / 3e-6 (24 nodes) of estT_fix on 30s seed 40 and 30sL2 seed 0, at +3-10% ops
([rulecost_30s_40.log](rulecost_30s_40.log), [rulecost_30sL2_0.log](rulecost_30sL2_0.log)). Variant B (24 nodes, p = 3) is
therefore the corrected value at a fair cost; A is the pre-registered value and an upper bound on the cost.

### 4-7. Note-only rows
- E45 (8,4) part, T133 audit, T108 diagnostics: saved old-rule (8,4) values are off by ~1.1e-3 (zoo0 .9750473 vs
  .9761608); E46 already supersedes them; T133's "nine H8T4 failures survive" should be marked superseded too.
- T110-T4: recipe-mode masses carry ~1e-4 old-rule error; its 2.39e-4 quadratic-vs-nearest difference is not
  attributable to the partition without a re-run on a fixed engine (runner pinned to an absent Codex workspace).
- E47 comparator column: unchanged in substance (ratios 18-161 -> about 1.1x in B).
- T192 GPU port: fixed engine, not affected.

## Verdicts

- **No rung verdict changes** in either variant: (4,4) fails on the zoo and on every T111 set; F1/F2 fail.
- **T111's single both-pass model** (30sL2 seed 0) **holds in B** (clean 0.18, noisy 0.96) but **not in A** (0.80 /
  4.25): its pass is real only if the mass rule is both correct and cheap. 30s seed 40: clean B 0.43 / A 1.87 (noisy
  fails in every variant). 30s seed 10: clean B 0.94 / A 4.08 (noisy fails).
- **T428 correction:** its "31s seed 0 becomes a clean pass (0.75)" used old ops with corrected truth; with a
  consistent correct engine it is 1.18 (B) / 4.98 (A), i.e. still a fail. Its 128-node truths equal estT_fix to 4e-9.
  Its root cause is refined: the p = 2 grading of the 12-node 'wide' rule, not the node count.

## Compute accounting

Single-core CPU; estT ops as counted by each engine (estT_fix counts its costlier orthant). Re-scoring: 20 models x
(exact + recipe + 8 noisy recipes), variant A ~1 h on 2 cores, variant B ~45 min; rulecost 2 x 6 rules x 2 modes ~40 min.

## Claims

- [measured] T192's GPU port and the t194/t195 estT copies use the fixed orthant; the old rule survives only in
  understand_m43/estT.py, its identical snapshots and class_region_families/estT_snapshot.py.
- [measured] Correcting the old rule changes no ladder-rung verdict (E41, T111); it changes recipe op counts by ~4.5x
  if estT_fix's mass rule is used (A) and by ~1.1x with a p = 3 graded 24-node rule (B).
- [measured, n = 2 models] The old rule's error comes from its p = 2 grading: a p = 3 graded rule with 12-24 nodes is
  within 5e-6 of estT_fix.
- [inferred] estT_fix's mass rule is ~4x more expensive than needed at (4,4); a cheaper validated rule would cut
  recipe and exact costs there by ~4x (not tested beyond two models; not tested at (8,4)/(8,5)).

## Alternatives not ruled out

- Variant B is post-hoc: its rule (24 nodes, p = 3) was chosen after seeing A. It is checked against estT_fix on two
  models only; on the others B's estimates agree with A's to ~1e-6 in recipe value, which supports it indirectly.

- estT_fix is itself validated at (8,4)/(8,5) (leaf sums 1 - 1e-11, |z| vs MC) and here against T428's 128-node
  rule on two (4,4) models; it is not independently validated on every model re-scored here.

## Failure modes

- Noisy truths are 1e7 MC (s.e. ~1.6e-4), so noisy ratios near 1 remain uncertain at that level.
