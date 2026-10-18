# gpu_cost_law: T192 (compiled/GPU exact reference, cost law, T5 go/no-go) and the T193 check of E46/E48

**Question.** What is the real cost law of exact enumeration and recipe108 (estT_fix, E46/E48) along T, and is an
exact M_16,10 answer key (T5) reachable with the owner's Colab budget (~305 units)? Secondary (DRI, for T193): an
independent re-check of E46/E48 (fresh noise seeds, own truth, own op count, MSE ratios).

**Status.** done for the decision; (16,5) seeds 1-4, (4,6), (8,6) and the (8,5) nominal audit still running.
**Verdict (unverified): T5 is NO-GO for exact enumeration.** The target's own prefixes have 34, 792, 21,032, 632,394
exact activation regions for k = 2..5 steps (growth x23, x27, x30 per step, increasing). k = 10 projects to
>= 1e13 regions, ~4 orders of magnitude beyond the whole budget even before cone masses (97% of exact ops at T = 4).

Colab how-to (login, runtime choice, transfer, reclaim-safe queues): [COLAB.md](COLAB.md).

## Method
Exact and recipe108 are deductive (no random model inputs); truths are plain Monte Carlo (sampling, diagnostic).
- `estT_base.py`: verbatim copy of `estT_fix/estT.py` (geometry, recipe, op counting untouched).
- `estG.py`: simplicial cones are buffered and integrated in batches with estT's formulas in torch float64 (CPU or
  CUDA); ops charged per simplex exactly as estT. Optional `T192_SLIVER=0.6`: thin "sliver" cones (tiny det but two
  rays far apart, where estT's ray-space thin rule is not converged) are bisected through the normalised midpoint of
  their widest ray pair (exact partition) until every pair has cos >= 0.6, then the thin rule is applied.
- `geomfast.py`: numba versions of `_adjacent_pairs` and the triangulation `faces_of` (85% of geometry time), same
  loops/order/rank rule/op charges.
- `estP.py`: parallel over the independent start cones; resumable per-cone logs. Options: `--start truth` (exact
  only: the 2T(T-1) truth cones {x_i > x_j > rest} x sign(x0) instead of the 4T! ordering half-chambers, DRI
  finding), `--leafsum` (integrate every leaf cell whole; sum must be 1), `--sample f` / `--nomass` (cost estimates),
  activation-region count (T194 convention, distinct leaf gate patterns) on every exact run.
- `orth6.py`: 6-D cone masses (T = 6): thin rule 6^5 / 8^5 by |det|, nested Plackett above |det| 1e-2.
- `prefix.py`: exact regions of the first k steps of M_16,10 (geometry only, simplex-fan start cut by x0 = 0).
- `audit.py`: T196 nominal FLOP audit of unchanged estT (per-function, LAPACK-standard SVD/LU/inverse costs).
- `mc_gpu.py`: 1e10-sample float64 MC on the GPU (no estT code); `export_simplices.py`: sample for Codex lane 34.
- `summary.py` -> `results/summary.md` (all tables), `results/summary.json`. Colab scripts in `colab/`.

## Results ([measured] unless tagged; all unverified until an independent check)
Full tables: `results/summary.md` (`python summary.py`). Key numbers:

**1. Port validation.** (8,4) seeds 0-4, exact and recipe108: ops, cells, resolved identical to E46, accuracy within
8e-12. (8,5) zoo0 exact: ops 312312258369, cells 203685, simplices 4295166 identical to E48, acc diff 3.2e-10.
Unchanged estT on Colab CPU, (8,5) zoo1 recipe108: ops/cells/resolved/simplices identical to the port, acc diff 6.6e-12
(`results/baseline_cpu/`). (16,3), (16,4) zoo0 match T194's independent counts (e.g. (16,4) recipe 25378 cells,
2.97e9 ops; exact 4.19e9 ops, 26742 regions). Command: `python estP.py H T seed exact|recipe108 11 cuda`.

**2. Truth (T193).** Exact vs two independent 1e10-sample MC streams per model: z mixed in sign (e.g. (8,4) seeds
0-4: +0.35/+0.70, +0.28/-1.56, +1.57/+0.88, -0.26/-0.09, +1.09/+1.49). The first-stream all-positive pattern was a
common-random-numbers artefact (one stream for all models). `python mc_gpu.py H T model 10 seed`.

**3. recipe108 vs sampling (T193).** (8,4): all 5 seeds and fresh-noise copies 1000-1003 win (worst MSE ratio
0.27 charged ops / 0.36 audited FLOPs, seed 4). **(8,5): seed 2 LOSES (ratio 3.4 charged / ~4.6 FLOPs), fresh copy
1000 ties (1.0 / ~1.4)**; the other 7 (8,5) models win. By wall clock on the same A100, GPU float64 sampling matches
or beats the recipe at (8,5) (ratios 0.1-130). Recipe saves only 1.2-1.5x ops over exact at T >= 4.

**4. Op completeness (T196).** Nominal audit (estT unchanged, every function wrapped, MA = 2, SVD values-only
4mn^2 - 4n^3/3, LU 2n^3/3, inverse 2n^3): complete / charged = 1.36-1.39 on 9 (8,4) runs (orthant quadrature
dominates); 1.38 on one (8,5) half. Caveats (review): nominal LAPACK formulas only; scalar/incidence work priced at
1 op per element; numba's SVD (vectors requested) and implementation overheads not priced. `python audit.py 8 4 s mode`.

**5. Exact-mode cost reductions.** Truth-cone start: same accuracy (within 1e-11 at (8,4), all seeds), identical
region counts, cells 0.58x, ops 0.67x at (8,4); 0.42x cells at (8,5). Masses are 97% of exact ops at T = 4.

**6. Mass accuracy (answer-key level).** Leaf sums: (8,4) 5e-12..1.4e-11; (8,5) zoo0 truth-start 1.4e-8, (16,5)
2.8e-8: estT's thin rule is not converged on sliver cones (6^5 vs 8^5 at d = 6: up to 1e-4 rel per simplex,
+7.7e-6 rel summed, one-signed). Sliver bisection: (8,5) leaf sums 1.4e-10..4.9e-10, zoo0 acc .83580573419
(old halves-start value .8358057328, old truth-start .8358057300: old T = 5 exact values good to ~4e-9). At d = 6 it
cuts the summed error 8e-8 -> 4e-9 vs a 10^5-node reference. Fixed-degree Grundmann-Moeller cubature (deterministic,
polynomial point count, 286 points at degree 7 for d = 10): median rel error 1e-11..1e-13 at d = 6 but fails on the
same slivers (worst 3e-2), so a d = 10 mass method needs sliver refinement too.

**7. Noisy copies (exact cost, T = 4).** 2 copies each at relative sigma .005/.01/.02 (x matrix RMS) and ARC absolute
.05 on (8,4) and (16,4) zoo0: cells 0.92-1.14x, regions 0.94-1.21x, ops 0.94-1.33x of clean; accuracy drops to
.62-.97. Noise does not blow up the region count at T = 4.

**8. Cost law (exact, truth start, 5 seeds).** Regions per model: (8,3) 236, (8,4) 3,160, (8,5) 42,322 (zoo0):
x13.4 per step. (16,3) 838, (16,4) 28,200, (16,5) 786,114 (zoo0): x34, x28. M_16,10 prefixes: x23, x27, x30.
Ops per exact region ~2.4e5 (8,4) to ~1.3e6 (16,5). Wall at (16,5) zoo0: 16 min on an A100 VM (geometry 7341 CPU-s).

**9. T5 go/no-go: NO-GO.** M_16,10 prefix regions (`python prefix.py k 12`): k = 2: 34, 3: 792, 4: 21,032,
5: 632,394 (all leaves' centroids reproduce their histories at k = 3, 4; 1e7-sample lower bounds 34/685/7386).
[inferred] At >= x30 per step, k = 10 has >= 1.5e13 regions; at the measured ~2 ms CPU per cell (k = 5) the geometry
alone is ~3e10 CPU-s, versus ~3e5 CPU-s for the whole Colab budget. [speculative] A route to M_16,10 needs a method
whose cost grows more slowly than the region count (merging histories), not faster hardware.

## Compute accounting
Colab units: 304.80 at login (2026-10-17 23:06 UTC) -> 291.8 at 02:00 UTC 2026-10-18 (~13 spent): A100 5.3 u/h,
L4 1.5 u/h, CPU 0.08 u/h. ~1.5 units were lost to a Colab reclaim of both GPU runtimes (01:10-01:28 UTC).
Ops = estT's m43.Ops counter (geometry + per-simplex mass charges; d = 6 charge `orth6.ops6`, same convention).
MC forward passes are diagnostics. Local sandbox: light CPU only (<= 2 processes, <= 8 GB virtual).

## Claims
- [measured] The port reproduces E46/E48 and unchanged estT op/cell counts exactly, accuracies within 1e-11..3e-10.
- [measured] Exact accuracies agree with independent MC (no systematic offset after independent streams).
- [measured] recipe108 loses to sampling at (8,5) seed 2 (and ties on one fresh copy): the (8,5) rung fails.
- [measured] estT's op count is ~1.37x below a nominal complete FLOP count at (8,4) (and one (8,5) half).
- [measured] T = 5 exact values from estT's thin rule carry ~1e-8 error; sliver bisection brings leaf sums to <5e-10.
- [measured] Noise up to ARC's sigma 0.05 does not increase exact-enumeration regions by more than 1.21x at T = 4.
- [measured] M_16,10 prefix regions grow x23, x27, x30 per step for k = 2..5.
- [inferred] Exact enumeration of M_16,10 (T5) is out of reach by >= 4 orders of magnitude.

## Alternatives not ruled out
- Prefix growth could slow after k = 5 (saturation of the running max / clock structure); k = 6 (capped, sampled)
  is the only further point. Merging histories that do not affect the answer (T188) could change the law entirely.
- The T = 4 noise insensitivity may not hold at larger T.
- Audit pricing is nominal; real implementation costs (numba SVD with vectors, Python overhead) are higher.
- orth6 accuracy was calibrated on (8,6) zoo0 simplices only.

## Failure modes
- Colab reclaims runtimes without notice: queues archive results after every run (`colab/runq.sh`).
- estT's thin rule (all d) is inaccurate on sliver cones (fixed by `T192_SLIVER`).
- `colab exec` replies time out (~60 s): long work must be detached and polled.
