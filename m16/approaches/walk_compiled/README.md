# walk_compiled (T393): wall-clock cost of a compiled minimal_pair walk vs an equally optimised sampler

↑ [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · frozen recipe: origin/claude/t10-minimal-pair `m16/approaches/minimal_pair/` @ a5063c7d · CPU audit: origin/claude/t379-minimal-pair-check

## Question
H-E16 (T10's frozen walk quadrature beats sampling at H16 T5) is scoped to FPE accounting: the numpy walk is 10-35x
slower than sampling in wall time. T379 estimated that a compiled walk would be ~1.5-3x slower than sampling per FPE,
untested. Does a compiled walk beat an equally optimised sampler in wall time **at equal MSE**, and where does the
time go?

## Pre-registration (written 2026-10-19, before any benchmark run)
- **Estimator under test:** T10's frozen `est` (estimator.py, core.py, walk.py, hom.py copied verbatim from a5063c7d
  into `frozen/`, sha256 asserted). Only `walk.walk` is replaced by a compiled implementation; everything else
  (strata, Sobol rule, face-point maps, two-stage Neyman allocation, cost charging) is the frozen code path.
- **Reproduction gate (must pass before any timing is reported):** on the 5 zoo H16 T5 seeds (0-4) and T379's 5 fresh
  models f100-f104 (retrained here with T379's script and seeds, since the checkpoints are not in this container;
  checked against T379's truths), at budgets 16k and 64k: compiled-walk `est` = frozen `est` to <= 1e-12 relative on
  the estimate and on the charged cost. Bitwise is not expected: the compiled matmul sums in a different order
  than BLAS. Also checked at T = 10 (zoo seeds 0-4). Failing this, the timing is not reported as the frozen
  estimator's.
- **Two compiled variants:**
  - `repro`: compiled walk, Student-t CDF evaluated by scipy (`stdtr`, as frozen) on the cell boundaries.
  - `fast`: same walk, closed-form integer-nu Student-t CDF in compiled code. Must also pass the 1e-12 gate.
- **Sampler, optimised to the same degree:** compiled (numba) forward pass + argmax check over iid N(0,1) inputs, float64,
  same thread count (1 thread for both; BLAS threads pinned to 1). Two input variants are timed: normals from numpy's
  Generator (ziggurat) inside the timed region, and normals precomputed (pure model cost). The walk is timed with and
  without its model-free part (Sobol points + face-point maps) precomputed; the fair comparisons pair "with RNG" with
  "with face maps", and "precomputed normals" with "precomputed face points".
  Also timed: plain numpy BLAS sampler (best of chunk sizes).
- **Measurements**, same models, T = 5 (10 models) and T = 10 (zoo seeds 0-4):
  - wall time of one `est` call at 16k and 64k FPE (median of >= 5 repeats, warm);
  - sampler wall time per FPE (large n), giving sampler time at 16k / 64k;
  - **equal-MSE ratio** R = t_walk / t_samp(n_eq), n_eq = p(1-p) / MSE_walk, with MSE_walk the mean over models of
    (estimate - truth)^2 minus the truth's variance (truth from >= 4.8e7 samples), and p the model accuracy;
    reported per model group and per budget; R < 1 means the walk wins in wall time at equal accuracy;
  - FLOPs: charged FLOPs of the walk (frozen cost x FPE FLOPs) vs counted FLOPs actually executed by the compiled
    walk (instrumented counters), and the sampler's FLOPs; rough memory traffic (bytes read/written per row-cell
    and per sequence) and whether each kernel is compute- or memory-bound;
  - time breakdown of the walk: face maps / matmul steps / root tests / CDF / Python driver.
- **Verdict line (pre-registered):** "the walk beats sampling in wall time" at a (T, budget) iff R < 1 on every
  model group for the `fast` variant with face maps timed. Otherwise the scope limit stands, with the measured ratio.
- Threats noted in advance: (1) 4 shared cores, so timings are noisy; runs are taken while no other heavy task runs
  in this container and repeated; (2) at T = 10 the walk is known to lose in FPE (T11/T380), so R > 1 there is
  expected; (3) the Python driver (Neyman stages, Sobol, chunking) is part of the frozen estimator; its share is
  reported, and a "driver-free" floor is also reported (not scored).

## Method
- `frozen/`: T10's estimator.py, core.py, walk.py, hom.py, verbatim from a5063c7d (estimator.py sha256 asserted in
  `models.py`). `est` is called unchanged; only `walk.walk` is swapped for `walk_nb.walk`.
- `walk_nb.py`: numba port of `walk.walk`, one row at a time (rows are independent in the frozen code; its global
  iteration cap is a per-row cell cap). Same operations in the same order except the H-length dot products, which
  sum left to right instead of in BLAS order. Modes: `repro` (emits the CDF arguments, evaluates the caller's scipy
  `stdtr` CDF on them in one numpy call, accumulates in the frozen order); `fast` (closed-form integer-nu Student-t
  CDF inside the compiled loop); `fastfm` (exploratory, not pre-registered: `fast` compiled with reassociation and
  FMA contraction allowed, inf/nan semantics kept, so the length-16 reductions vectorise). Instrumented FLOP counters.
- `sampler.py`: compiled sampler (numba, float64, blocks of 64 sequences laid out for SIMD across sequences; first-max
  argmax as numpy), with normals from numpy's Generator inside the timed region (`compiled_rng`) or precomputed
  (`compiled_pre`); an exploratory fastmath copy (`compiledfm_pre`); and the plain numpy/BLAS sampler (`numpy`).
  `python sampler.py truth`: 1e8-input truths per model (`truths.json`).
- `models.py`: zoo seeds 0-4 (H16, T5 and T10); fresh models f100-f104 retrained with T379's script and seeds
  (`train_fresh.py`, output dir changed only). They are T379's models: frozen `est` gives T379's recorded estimates
  bitwise (f100 16k 0.9901486356839357, f104 16k 0.9919287070023654), and the truths agree within 1.5 se.
- `repro.py T`: the reproduction gate. `bench.py T`: wall times (one thread, BLAS pinned to 1, median of 7 warm
  repeats; frozen numpy: 3). `analyse.py`: equal-MSE ratios -> `analysis.json`, `analysis.log`.
- Model-free precompute (`_pre` rows): Sobol rules and face-point maps memoised across `est` calls (they do not depend on
  the model; T10 already charges them 0 FPE). The sampler counterpart is precomputed normals.

## Status
Done (2026-10-19).

## Results

### Reproduction gate (pre-registered): PASS
`python repro.py 5` -> `repro_T5.json`; `python repro.py 10` -> `repro_T10.json`.

| set | budgets | worst relative difference, estimate or charged cost (repro / fast / fastfm) |
|---|---|---|
| T5: zoo 0-4 + f100-f104 | 16k, 64k | 1.1e-16 / 1.1e-16 / 2.3e-16 (most rows bitwise) |
| T10: zoo 0-4 | 16k, 64k | 1.3e-15 / 1.3e-15 / 1.3e-15 |

### Wall time at equal MSE (`python bench.py 5; python bench.py 10; python analyse.py`)
One thread, 4-core container with no other job running. Times are means over the group's 5 models of the per-model
median. R = t_walk / t_samp(n_eq), where t_samp(n_eq) is the time the sampler needs to reach the walk's MSE. To first
order, R = (walk / sampler time per FPE) x (MSE ratio at equal FPE). The bracket is a bootstrap over the 5 models
(it carries only the MSE-ratio uncertainty, since the walk is deterministic).

Pre-registered pairing: `fast` est (face maps, Sobol and Python driver all timed) vs compiled sampler with RNG timed.

| T | group | FPE | walk est | sampler, same FPE | per-FPE time ratio | MSE ratio [95%] | **R at equal MSE [95%]** | exploratory fastfm R |
|---|---|---|---|---|---|---|---|---|
| 5 | zoo | 16k | 29.3 ms | 8.6 ms (526 ns/FPE) | 3.39 | 0.087 [0.037, 0.163] | **0.30 [0.13, 0.55]** | 0.21 |
| 5 | zoo | 64k | 88.4 ms | 33.7 ms | 2.63 | 0.062 [0.008, 0.162] | **0.16 [0.02, 0.43]** | 0.10 |
| 5 | fresh | 16k | 27.0 ms | 8.6 ms (522 ns/FPE) | 3.13 | 0.046 [0.007, 0.095] | **0.15 [0.02, 0.30]** | 0.10 |
| 5 | fresh | 64k | 75.6 ms | 33.4 ms | 2.27 | 0.039 [0.009, 0.078] | **0.09 [0.02, 0.18]** | 0.06 |
| 10 | zoo | 16k | 47.7 ms | 16.3 ms (1024 ns/FPE) | 2.93 | 1.06 [0.10, 2.49] | **3.1 [0.30, 7.3]** | 2.1 |
| 10 | zoo | 64k | 146 ms | 65.5 ms | 2.23 | 0.38 [0.18, 0.64] | **0.84 [0.40, 1.43]** | 0.57 |

The other pairings are in `analysis.log`:
- Model-free part precomputed on both sides: the per-FPE ratio is 2.1-2.7 at T5 and 2.3-2.4 at T10.
- Walk with scipy CDF (`repro`): per-FPE ratio 3.5-4.5.
- Frozen numpy walk vs numpy sampler: per-FPE ratio 18-39 at T5 (T379 measured 10-35). At equal MSE, R = 0.70-3.4 at T5.

**Verdict (pre-registered line: R < 1 on every group for `fast` with face maps timed):**
- **T = 5, 16k and 64k: yes.** The compiled walk beats an equally compiled sampler in wall time at equal MSE, by
  3.4x-11x (R = 0.09-0.30). Every bootstrap upper bound is <= 0.55.
- **T = 10, 16k: no** (R = 3.1).
- **T = 10, 64k: R = 0.84 nominally**, which meets the pre-registered line. But the CI crosses 1 [0.40, 1.43], and the
  MSE ratio rests on 5 deterministic draws, one per model; the per-model squared-error ratios are 0.06-0.88.
  This is not a robust win. The T10 FPE-accuracy question belongs to T11/T380, not this task.

### Where the time goes (T5, zoo, 16k, `fast`)
- **Total: 29.3 ms.**
  - Compiled walk: 19.4 ms.
  - Sobol rule generation and face-point maps (model-free, numpy/scipy): ~7.6 ms.
  - Python driver (Neyman stages, chunking, concatenation): ~2.3 ms.
  - At 64k the walk is 81% of the total (71.5 of 88.4 ms).
- **CDF.** Replacing scipy `stdtr` by the closed form cuts the in-walk time by a third (29.1 -> 19.4 ms at 16k,
  101 -> 71 ms at 64k).
- **FLOPs.** The compiled walk executes 0.95x the FLOPs it is charged. Matrix-vector work is 0.79-0.80 of the charge;
  the rest is O(H) updates, gate tests, root tests and correct-set tests. So the charged FPE count is honest, as T379
  found.
- **Throughput.** The walk runs at 2.4-2.8 GFLOP/s (fastfm 4.0-4.7) against the sampler's 6.1-6.5 GFLOP/s.
  - Sampling processes 64 sequences in lockstep, so its inner loops vectorise across sequences.
  - The walk's rows diverge: different cell counts and restart steps. Each recomputed step is a 16x16 matrix-vector
    product per row, whose 16-term reductions are latency-bound without reassociation.
  - This throughput gap, not uncharged work, is the residual 2.2-3.4x per-FPE ratio.
- **Memory traffic** [inferred from data sizes]. Both kernels are L1-resident and compute/latency bound.
  - Walk: the per-row state (Z, DZ: 2 x T x H doubles = 1.3 KB at T5, 2.6 KB at T10) and the weights (2.3 KB) stay
    in L1; about 2.5 KB of L1 read+write per cell. DRAM traffic is about 100 B per face point (y, a, b, tau, weight)
    over several cells.
  - Sampler: 8T bytes of input per sequence, about 90 MB/s at T5, far below DRAM bandwidth.
  - Neither is memory-bound, so a GPU or wider SIMD would favour the sampler (it vectorises trivially) more than the
    walk.

## Compute accounting
- Unit: FPE = one forward pass of one length-T sequence = T(2H^2 + 2H) FLOPs (the frozen walk's charge unit).
- Truths: 15 models x 1e8 sequences, compiled sampler (~1 min per T5 model, ~2 min per T10 model).
- Retraining f100-f104: 5 x 2^30 sequences (T379's recipe, ~38 CPU min per model).
- Reproduction gate: 30 (model, budget) pairs x 4 `est` runs. Benchmark: per (model, budget) 6 variants x 8 runs,
  plus 4 samplers x 3 sizes x 8 runs; total about 1.5e8 FPE, about 20 CPU min.

## Claims
- [measured] The numba walk reproduces T10's frozen `est` to <= 1.3e-15 relative (estimates and charged costs, most
  rows bitwise) on zoo H16 T5/T10 seeds 0-4 and T379's fresh models f100-f104, at 16k and 64k.
- [measured] At H16 T5, at equal MSE, the frozen estimator with the compiled walk takes 0.09-0.30x the wall time of a
  compiled float64 sampler on one thread (zoo and fresh groups, 16k and 64k).
- [measured] Per FPE, the compiled walk is 2.3-3.4x slower than the compiled sampler (1.5-2.5x with fastmath, 2.1-2.7x
  with the model-free part precomputed). T379's 1.5-3x estimate holds.
- [measured] The walk executes 0.95x its charged FLOPs; the per-FPE gap is throughput (2.4-2.8 vs 6.1-6.5 GFLOP/s).
- [measured] At T = 10 the walk loses at 16k (R = 3.1). [inferred] 64k is inconclusive (R = 0.84 [0.40, 1.43]).
- [inferred] Removing the H-E16 scope limit "FPE accounting only" is justified at T = 5 for single-thread CPU. It is
  not justified for SIMD-wide or GPU hardware, which was not tested.

## Alternatives not ruled out
- **A stronger sampler.** Possible improvements: a faster normal generator (numpy's ziggurat costs 7-14% of
  `compiled_rng`), float32, AVX-512, multithreading. Precomputing the normals (which removes the RNG cost entirely)
  leaves R < 0.3 at T5. A 2x faster sampler kernel would still leave R <= 0.6 at T5, since the MSE gain is 6-25x.
- **A stronger walk.** Batching rows by restart step, or vectorising across rows, would lower R further; not tried.
- **The bootstrap over 5 models** is a rough CI. The T5 conclusion holds at every upper bound; T10 needs more models.
- **Timing noise.** A single benchmark run, though the earlier smoke test under load gave the same numbers within ~10%
  (z0 T5: 31/86 ms vs 29/88 ms). Not repeated on another machine.
- **Wall time includes the frozen Python driver.** A full compiled re-implementation would gain at most the ~2-3 ms
  driver share.

## Failure modes
- The `fast` CDF is the closed form for integer nu = T - 1 only (the radial integrand). The non-radial ablation path
  would need the Gaussian CDF.
- The CDF in `repro` mode is whatever vectorised function the caller passes. Bitwise equality also depends on the
  frozen walk's BLAS order agreeing with the left-to-right sum, which held in almost all rows, not by construction.
- Multi-thread or GPU comparisons were not made. The sampler parallelises trivially; the walk parallelises over rows
  but with divergent work.
