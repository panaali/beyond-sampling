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
(to be filled)

## Status
Running (pre-registered 2026-10-19).

## Results
(none yet)

## Compute accounting
(to be filled)

## Claims
(none yet)

## Alternatives not ruled out
(to be filled)

## Failure modes
(to be filled)
