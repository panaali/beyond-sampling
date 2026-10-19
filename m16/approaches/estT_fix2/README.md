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

Running (2026-10-19).

## Results

Pending.

## Compute accounting

Single-core CPU; estT ops as counted by each engine (estT_fix counts its costlier orthant).

## Claims

None yet.

## Alternatives not ruled out

- estT_fix is itself validated at (8,4)/(8,5) (leaf sums 1 - 1e-11, |z| vs MC) and here against T428's 128-node
  rule on two (4,4) models; it is not independently validated on every model re-scored here.

## Failure modes

- Noisy truths are 1e7 MC (s.e. ~1.6e-4), so noisy ratios near 1 remain uncertain at that level.
