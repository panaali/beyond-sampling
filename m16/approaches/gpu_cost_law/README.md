# gpu_cost_law: T192 (compiled/GPU exact reference and cost law; T5 go/no-go)

**Question.** What is the real cost law of exact enumeration and recipe108 (estT_fix, E46/E48) along T at H = 8, and
is an exact M_16,10 answer key (T5) reachable with the owner's Colab budget (~305 units)? Secondary (DRI request, for
T193): an independent re-check of E46/E48 with fresh noise seeds, an independent truth and MSE ratios.

**Status.** running. S0 (Colab smoke) and S1 (port validation, (8,4) all seeds, (8,5) zoo0) done; (8,5) seeds 2-4,
fresh-noise copies and the T = 6 stage queued on an A100.

## Method
Colab connection and runtime choice: [COLAB.md](COLAB.md).
Deductive (no random model inputs) for exact and recipe108; the independent truth is plain Monte Carlo (sampling).
- `estT_base.py`: verbatim copy of `estT_fix/estT.py` (geometry, recipe and op counting untouched).
- `estG.py`: every simplicial cone estT would integrate one at a time is buffered and integrated in batches with the
  same formulas (cofactor normals, atan2 angles, log-node Plackett, thin rule below |det| 1e-6), in torch float64 on
  CPU or CUDA. No recipe decision depends on a mass, so cells/splits/ops are unchanged; ops are charged per simplex
  exactly as estT charges them.
- `geomfast.py`: numba versions of the two geometry hot spots (`_adjacent_pairs`, triangulation `faces_of`; 85% of
  (8,5) geometry time), same loops, order, rank rule and op charges.
- `estP.py`: parallel over the 2 T! true-boundary halves (independent sub-problems): CPU workers run the geometry,
  the parent integrates each half's simplices on the GPU and logs one JSON line per half (resumable). Options:
  `--sample f` (seeded random fraction of halves; estimated totals with SE) and `--nomass` (geometry only).
- `orth6.py` (new, needed for T = 6): 6-D simplicial-cone masses. |det| < 1e-6: estT's ray-space thin rule
  (Duffy 6^5); 1e-6 <= |det| < 1e-2: same rule with 8^5 nodes; |det| >= 1e-2: nested Plackett (outer path
  R(t) = I + t(C - I), inner = estT's d = 4 form on the conditional correlation, partial correlations from the
  4x4-block cofactors as estT._orth5). The ray rule is evaluated as one GEMM (|R^T u|^2 = u^T R R^T u).
- `mc_gpu.py`: independent truth, 1e10 forward passes per model in float64 on the GPU (shares no code with estT).
- `summary.py` -> `results/summary.json` and the table below. Colab plumbing in `colab/` (`sync.sh` pulls results).
- Colab is driven from this container with the official `google-colab-cli` (OAuth, `colab new/upload/exec/download`).
  Code goes up as a tarball of `m16/` (repo is private; no token on the VM); runs are detached and polled.

## Results
**S0** [measured]: unchanged estT on Colab CPU reproduces (8,4) zoo0 (ops/cells identical, acc diff 7.9e-12):
`colab/s0.py` -> `results/s0_colab_cpu_84_zoo0.json`.

**Port validation (controls)** [measured], `python estP.py 8 T s exact|recipe108 11 cuda` on an A100 (12 vCPU):
- (8,4) seeds 0-4, exact and recipe108 (10 runs): ops, cells, resolved identical to E46 (`estT_fix/rescore/*.json`);
  accuracy differs by -4.3e-12 .. -8.1e-12. Local CPU port: 5.4e-14 (zoo0 exact).
- (8,5) zoo0 exact: ops 312312258369, cells 203685, cut 147976, resolved 203277, simplices 4295166, all identical to
  E48 (`estT_fix/run5/zoo0_exact.json`); acc .8358057328 vs .8358057331 (diff 3.2e-10). Wall 387 s vs 2622 s (E48).
- Mass routines vs estT.orthant per simplex: d = 4: 3.3e-13, d = 5: 1.5e-13 (GPU), identical op charges.
- orth6: stellar-subdivision identity (parent = sum of 6 sub-cones) holds to <= 5e-14 abs on wide cones, 3e-15 abs
  (1e-7 rel) across the thin/wide seam, 1e-15 rel on thin; 64 sign-orthants sum to 1 within 1e-15; vs 4e9-sample MC on
  3 wide cones z = +2.0, +1.0, +0.4. Ray rule 8^5 vs nested Plackett on the 120 widest of 1.15M real (8,6) zoo0
  simplices: max rel 5.7e-7, summed abs 1.3e-12 (6^5: 4.5e-5 / 1.4e-10).

**Table** (`python summary.py`; err = recipe108 - exact; MC = 1e10 samples, z = (MC - exact)/se):

| (H,T) | model | exact acc | MC z | recipe err pp | ops (exact / recipe) | cells (exact / recipe) | wall s (exact / recipe) | MSE ratio ops / wall |
|---|---|---|---|---|---|---|---|---|
| (8,4) | 0 | 0.976160808 | +0.35 | -0.000013 | 1.1e+09 / 8.87e+08 | 8455 / 6182 | 4 / 4 | 9.7e-07 / 0.00039 |
| (8,4) | 1 | 0.987460011 | +0.28 | -0.000568 | 9.64e+08 / 6.28e+08 | 8447 / 4573 | 4 / 3 | 0.0025 / 1.1 |
| (8,4) | 2 | 0.971852150 | +1.57 | -0.007253 | 1.18e+09 / 8.63e+08 | 9470 / 6145 | 5 / 4 | 0.25 / 1e+02 |
| (8,4) | 3 | 0.976713497 | -0.26 | -0.000001 | 1.1e+09 / 9.23e+08 | 9371 / 6799 | 5 / 5 | 2.2e-09 / 9.3e-07 |
| (8,4) | 4 | 0.985397918 | +1.09 | -0.004905 | 1.35e+09 / 1.07e+09 | 11312 / 7679 | 5 / 5 | 0.27 / 1.1e+02 |
| (8,5) | 0 | 0.835805733 | +0.97 | -0.000226 | 3.12e+11 / 2.18e+11 | 203685 / 114155 | 387 / 438 | 0.0097 / 1.6 |
| (8,5) | 1 | 0.864562575 | +1.03 | -0.000172 | 2.61e+11 / 1.89e+11 | 164541 / 98580 | 556 / 431 | 0.0057 / 1 |

(rest of (8,5), fresh-noise copies and T = 6 pending)

## Independent check of E46/E48 (for T193)
Different implementation of the masses (batched torch, GEMM), different hardware, own truth by 1e10-sample GPU MC,
fresh noise seeds 1000-1003 (`leaderboard._noisy(zoo0, 1, .005, k)`). Pending completion; partial: E46 exact
accuracies and recipe108 errors reproduced on all 5 seeds at (8,4), E48 at (8,5) zoo0; exact agrees with MC
(|z| <= 1.6, 7 models). Caveat: the geometry is the same code (estT_base), so this is not an independent check of the
recipe logic or of the cell enumeration, only of the masses, the op count reproduction and the truth.

## Compute accounting
Colab units: 304.80 at login (2026-10-17 23:06 UTC); 301.35 at 23:58 (A100 session 5.30 u/h + CPU session 0.08 u/h).
Ops = m43.Ops counter of estT (geometry + per-simplex mass charges; for d = 6 a new charge `orth6.ops6`, same
convention). MC forward passes are diagnostics only (not charged to any estimator).

## Claims
- [measured] The port reproduces E46 ((8,4), 5 seeds, both modes) and E48 ((8,5) zoo0 exact) op and cell counts
  exactly and accuracies within 1e-11 / 3.2e-10.
- [measured] Exact (8,4)/(8,5) accuracies agree with independent 1e10-sample MC (|z| <= 1.6, 7 models).
- [measured] On an A100 VM the (8,5) exact run takes 6.5 min wall (11 CPU workers + GPU masses), ~7x faster than the
  CPU reference; geometry (CPU) is the bottleneck, GPU utilisation ~16%.
- [measured] Wall-clock comparison on the same A100: recipe108 vs float64 MC gives MSE ratios 1.0-1.6 at (8,5)
  (2 models) and up to ~100 at (8,4) seeds 2/4: by wall time the recipe does not beat GPU sampling.

## Alternatives not ruled out
- The 6 of 7 positive MC z values (exact slightly below MC by ~1e-6) could be a small positive bias of the error masses
  or chance: MC used one random stream per T for all models (common random numbers), so the z's are correlated.
  Check: re-run MC with different seeds.
- The 8e-12 GPU/CPU accuracy offset is attributed to float64 det/atan2 differences; not isolated.
- orth6 was calibrated on (8,6) zoo0 simplices; other models may have wider simplices (Plackett path, slower).

## Failure modes
- Colab session reclaim kills detached runs (per-half logs make runs resumable).
- `colab exec` replies time out (~60 s), so everything long must be detached.
- orth6 nested Plackett is slow (~200 simplices/s); a model with many |det| >= 1e-2 simplices would be expensive.
