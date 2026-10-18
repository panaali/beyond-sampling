# gpu_cost_law: T192 (compiled/GPU exact reference, cost law, T5 go/no-go), T193 check of E46/E48, T222 coverage

**Question.** What is the real cost law of exact enumeration and recipe108 (estT_fix, E46/E48) along T, and is an
exact M_16,10 answer key (T5) reachable with the owner's Colab budget (~305 units)? Secondary: the T193 re-check of
E46/E48 (fresh noise, own truth, own op count, MSE ratios), the T196 op-completeness audit, the T222 mass-coverage
curve of the target's prefix regions, and a high-precision MC truth for M_16,10.

**Status.** done (wind-down). Pending: (8,5) nominal FLOP audit (CPU runtime), final MC/k=6 numbers (see Results 10-11).
**Verdict (unverified; DRI decision): T5 NO-GO.** Scope: region-by-region exact enumeration plus per-region cone
masses (estT exact mode and its truth-cone variant) on M_16,10. Its prefixes have 34, 792, 21,032, 632,394 exact
activation regions for k = 2..5 steps (growth x23, x27, x30 per step, increasing), so k = 10 projects to >= 1e13
regions, >= 4 orders of magnitude beyond the budget. Not excluded: methods that aggregate across regions (merging
histories, T188) or mass-ordered truncation (T222: N(1-1e-6) grows ~x17/step, still ~2e11 regions at k = 10).

Colab how-to (OAuth login, runtime choice and rates, transfer, reclaim-safe queues): [COLAB.md](COLAB.md).

## Method
Exact and recipe108 are deductive (no random model inputs); every truth/check by MC is sampling (diagnostic only).
- `estT_base.py`: verbatim copy of `estT_fix/estT.py`. `estG.py`: cone masses buffered and integrated in batches
  with estT's formulas in torch float64 (CPU/CUDA), ops charged exactly as estT; `T192_SLIVER=0.6` bisects thin
  sliver cones (exact partition) before estT's thin rule. `geomfast.py`: numba `_adjacent_pairs` / triangulation.
- `estP.py`: parallel over start cones, resumable; `--start truth` (exact only; 2T(T-1) truth cones instead of 4T!
  half-chambers), `--leafsum`, `--sample`, `--nomass`; region counts (T194 convention) on exact runs.
- `orth6.py` (6-D masses), `gmass.py` (masses summed per group), `prefix.py` (M_16,10 prefix regions and masses),
  `audit.py` (T196 nominal FLOPs), `mc_gpu.py` / `mc_target.py` (GPU MC truths), `export_simplices.py` (Codex 34),
  `summary.py` -> `results/summary.md` (all tables). Colab queues and helpers in `colab/`.

## Results ([measured] unless tagged; unverified until an independent check)
**S0/S1 port validation.** (8,4) seeds 0-4 exact+recipe108: ops/cells/resolved identical to E46, acc within 8e-12.
(8,5) zoo0 exact: ops 312312258369 / cells 203685 / simplices 4295166 identical to E48, acc diff 3.2e-10. Unchanged
estT on Colab CPU, (8,5) zoo1 recipe108: identical counts, acc diff 6.6e-12. (16,3), (16,4) zoo0 identical to T194.
`python estP.py H T seed exact|recipe108 11 cuda`.

**S2 (8,5) all seeds and T193 check** (`results/summary.md`, main table; cost columns: estT charged ops / charged
x1.37 / audited nominal FLOPs; MSE ratio by charged ops, by FLOPs (682 / 853 per sample, T196) and by A100 wall):
- Truth: exact vs two independent 1e10-sample MC streams per model, z mixed in sign (first-stream all-positive
  pattern was a common-random-numbers artefact).
- (8,4): all 5 seeds and fresh copies 1000-1003 win (worst 0.27 charged / 0.36 audited, seed 4).
- **(8,5): seed 2 loses (3.4 charged / ~4.6 FLOPs), fresh copy 1000 ties (1.0 / ~1.4)**; 7 other models win.
- **(16,5) zoo0: recipe108 .9900760545 vs exact (truth start) .9901084177: err -3.2e-5, MSE ratio 34.5 (charged).**
- By wall clock on the same A100, GPU float64 MC beats the recipe at (8,5) (ratios 0.1-130).

**S3 cost law** (exact, truth start, `results/summary.md` cost-law table):
| (H,T) | regions (5 seeds, mean) | cells | ops | growth/step |
|---|---|---|---|---|
| (8,3) | 236 | 331 | 2.6e5 | |
| (8,4) | 3,160 | 5,510 | 7.7e8 | x13.4 |
| (8,5) | 42,322 (zoo0) | 84,735 | 1.6e11 | x13.4 |
| (16,3) | 838 | 1,020 | 1.1e6 | |
| (16,4) | 28,200 | 38,100 | 3.4e9 | x34 |
| (16,5) | 786,114 (zoo0) | 1,147,093 | 1.0e12 | x28 |
Throughput ((16,5) run, A100 VM): geometry 4.2e6 counted ops/s per CPU core (3.2e7 on 11 workers); GPU masses
~7.7e9 counted ops/s (7.1e4 5-D simplices/s incl. sliver handling). Masses are 97% of exact ops at T = 4; wall time
is geometry-bound at T >= 5. Truth-cone start: same accuracy, identical regions, 0.58x cells, 0.67x ops at (8,4).

**M_16,10 prefixes (`python prefix.py k 12`)**: regions 34 / 792 / 21,032 / 632,394 for k = 2..5 (cells 37 / 894 /
24,407 / 748,585; geometry ops 9.4e3 / 7.8e5 / 7.6e7 / 9.9e9). Leaf centroids reproduce recorded histories (k = 3, 4);
1e7-sample distinct-pattern lower bounds 34 / 685 / 7,386.

**T222 mass coverage (`python prefix.py k w 0 masses cuda`, npz in `results/prefix/`)**: N(delta) = regions covering
1 - delta of the mass (leaf sums 1 within 9e-14 / 2e-11 / 1.4e-9; k = 3 masses vs 2e7 samples chi2/dof 1.03;
k = 3, 4 identical to the T194 child's independent T222 numbers):
| k | 1e-1 | 1e-2 | 1e-3 | 1e-4 | 1e-5 | 1e-6 | effective (inv. Simpson) |
|---|---|---|---|---|---|---|---|
| 3 | 66 | 201 | 388 | 527 | 634 | 708 | 37 |
| 4 | 396 | 1,737 | 4,018 | 7,021 | 10,233 | 13,164 | 112 |
| 5 | 2,548 | 14,658 | 42,499 | 89,216 | 152,956 | 226,080 | 397 |
Per-decade factors rise with k (last decade 1.12 / 1.29 / 1.48).

**Noisy copies (exact cost at T = 4)**: relative sigma .005/.01/.02 and ARC absolute .05, 2 copies each, (8,4) and
(16,4) zoo0: cells 0.92-1.14x, regions 0.94-1.21x, ops 0.94-1.33x of clean (accuracy .62-.97): no blow-up.

**Mass accuracy.** estT's thin rule is not converged on sliver cones: (8,5) leaf sum 1 + 1.4e-8, (16,5) 1 + 2.8e-8
(old T = 5 exact values good to ~4e-9). Sliver bisection: (8,5) leaf sums 1.4e-10..4.9e-10, zoo0 acc .83580573419.
d = 6 (orth6): subdivision identity <= 5e-14; sliver fix cuts summed error 8e-8 -> 4e-9. Grundmann-Moeller fixed
cubature (polynomial point count, deterministic) reaches 1e-11..1e-13 median error at d = 6 but needs the same
sliver refinement (worst 3e-2), which a d = 10 mass method would also need.

**Op completeness (T196).** Nominal audit, 9 (8,4) runs: complete / charged 1.36-1.39 (one (8,5) half: 1.38).
Nominal only: LAPACK formulas (SVD values-only), scalar/incidence work at 1 op per element, numba SVD vectors and
implementation overhead not priced. (8,5) seeds 0-4 audit: pending (`colab/q_audit.sh`).

**M_16,10 MC truth (`python mc_target.py 300`)**: float64 forward, 300 blocks of 1e9 samples, Philox seed
20261018000 + block (all block counts in `results/mc_target/blocks.jsonl`; extendable to 1e12 by running more blocks):
**acc 0.9531174789 +- 3.9e-7** (3e11 samples; between-block SD of the mean 3.4e-7, consistent with binomial), vs
leaderboard_truths orig .95313575 +- 3.7e-5 (z -0.49) and m16/truth.txt .95317945 (2e7 samples, z -1.31).
float32 forward on block 0: +29 correct of 1e9 (2.9e-8).

**M_16,10 prefix k = 6 (capped 40 min, seeded random subtree sample, with masses)**: 245 of 3,811 subtrees:
1,473,541 cells -> **estimated 2.3e7 cells at k = 6 (x30.6 over k = 5)** [inferred, unbiased subtree estimate];
1,459,727 distinct regions in the sample (lower bound) covering 8.8% of the mass, so its N(delta) is not meaningful.
Region growth stays ~x30 per step through k = 6 [inferred: cells ~ regions].

## Compute accounting
Colab units: 304.80 at login (2026-10-17 23:06 UTC) -> 282.58 at 03:22 UTC (22.2 spent): A100 5.3 u/h, L4 1.5 u/h,
CPU 0.08 u/h. Four GPU runtimes were reclaimed by Colab without notice (01:10-01:28 both, ~02:10 L4, ~02:40 prefix
L4, ~03:10 A100); work lost ~4 units. Ops = estT's m43.Ops counter (+ `orth6.ops6` at d = 6). MC is diagnostic.

## Claims
- [measured] The port reproduces E46/E48/T194 and unchanged-estT op/cell counts exactly; accuracies within 3e-10.
- [measured] Exact accuracies agree with independent MC streams (no systematic offset).
- [measured] recipe108 loses to sampling at (8,5) seed 2 (3.4x) and at (16,5) zoo0 (34.5x, charged ops).
- [measured] estT's charged ops are ~1.37x below a nominal complete FLOP count at (8,4).
- [measured] estT's thin rule leaves ~1e-8 errors at T = 5; sliver bisection brings leaf sums below 5e-10.
- [measured] Noise up to ARC's sigma .05 changes T = 4 regions by <= 1.21x.
- [measured] M_16,10 prefix regions grow x23, x27, x30 per step (k = 2..5); N(1 - 1e-6) grows x18.6, x17.2.
- [inferred] Region-by-region exact enumeration of M_16,10 needs >= 1e13 regions (>= 2e11 with mass-ordered
  truncation at 1e-6): out of reach by >= 4 orders of magnitude (T5 no-go, scoped as above).
- [speculative] An exact route needs a method whose cost grows slower than the region count (history merging).

## Alternatives not ruled out
- Prefix growth might slow after k = 6 (k = 6 is a capped-sample estimate).
- T = 4 noise insensitivity may not hold at larger T; orth6 calibrated on (8,6) zoo0 only.
- Audit pricing is nominal; the real implementation cost is higher.
- (16,5) recipe compared against the truth-start exact value (the halves-start exact run was lost to a reclaim).

## Failure modes
- Colab reclaims runtimes without notice: archive every run (`colab/runq.sh`) and sync often.
- estT's thin rule (all d) is inaccurate on sliver cones (fixed by `T192_SLIVER`).
- `colab exec` replies time out (~60 s): long work must be detached and polled.
