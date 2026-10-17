# Incremental exact segment walker (T20, first estimator port)

## Question

Can the prefix reuse charged by the original `sci_tao` actually be implemented,
while retaining the same line integrals and honestly accounting for root,
readout, and integration work across hidden sizes and sequence lengths?
This is a numerical building block and a quadrature baseline, not a proposed
mechanistic solution of the ARC challenge.

## Method

`core.py:walk_mass` has the original interface and returns `(mass, cost, cells)`.
With `details=True` it also returns per-line operation counters. The weights are
`(Wi[H], Wh[H,H], Wo[T,H])`; all H >= 1 and T >= 2 are supported by the code.

For each visited cell, cache both affine coefficients of every hidden state:
`h_s(t) = B_s + t*S_s`. When a gate first switches at step k, only steps k onward
are recomputed. Earlier hidden states and their future gate roots are retained.
The input-independent prefix before the first moving coordinate gets one value
pass and no tangent pass. Each remaining linear piece is integrated by clipping
the correct-output inequalities and taking Gaussian CDF differences. Empty
half-intervals do not trigger output-envelope work. The Gaussian face geometry
and Sobol rules are imported unchanged from the original folder.

`estimator.py` exposes the shared `arc_eval` / leaderboard
`est(model, budget, rng) -> (estimate, actual_cost)` interface. The fixed Sobol
seed and pilot heuristic are inherited from the original implementation; no
constants were fitted. The pilot is reused and its allocation arithmetic counted.
`ADVICE=None`, `PRECOMPUTE_FLOPS=30_000_000`, and explicit `SUPPORTS` are declared.
The shared loader currently accepts a finite support list, so the declaration
lists the usual zoo grid plus tested random sizes; the function itself is generic.
The DRI must classify the new entry as **quadrature**; it is currently unreviewed
by the shared leaderboard. Do not apply the legacy `sci_tao` 2.23 cost override.

Two alternative explanations tested during validation:

1. A reported saving could still be fictitious prefix accounting. Check the
   executed prefix/suffix counters against full-recompute cell counts; every
   suffix count corresponds to two actual dense recurrent products.
2. Agreement with the old walker could preserve an existing integration bug.
   Compare to dense independent forward grids, exactly solvable zero-logit and
   tiny constant-gap examples, and force the cell-limit failure path.

## Status

Done for this bounded `sci_tao` port, including an optional compiled backend.
Other T20 estimator ports remain open.
No leaderboard test-model accuracy evaluation or tuning was performed. The
frozen DEV runs do not establish competitive MSE. Independent review is pending.

## Results

**Correction after independent review:** commits before the root-fix follow-up
can retain a stale prefix if the positive probe crosses an earlier cached gate
root. The original 2,112 ordinary-case checks did not catch this. Both backends
now invalidate every crossed cached prefix and select right-hand slopes at
exact roots, including eps=0. They also reject nonfinite intermediate states.
`root_regression.json` contains 108 closed-form cases spanning near-coincident
and simultaneous roots, three gap sizes, two window choices, and zero/default
epsilon; maximum error is 5.56e-17. The reviewer's independent checks covered
42 closed-form cases and 320 random lines, with 20 independently gridded lines;
the root blockers were resolved. Run `python
m16/approaches/sci_tao_v2/check_roots.py` after rebuilding to reproduce.

The extra root checks, right-hand gates, and finite-state guards are charged.
**The earlier `validation.json` and `dev_benchmark.json` below are retained
historical pre-fix snapshots, not current costs or budget-allocation results.**
Current backend parity, timings, and six frozen DEV estimates are in the
regenerated `compiled_validation.json`; no new truth samples were taken.
At requested 16k, current estimates/costs are .940815101/16485.037 for DEV sigma
.0025 and .628705085/16237.861 for DEV sigma .02. Changed costs alter face counts
under the same frozen pilot rule; no constants were tuned. The poor latter
estimate is retained as a negative result. Current C++ CPU time is .0309/.0315
seconds, 5.79x faster than Python; warm-batch speedup over legacy is 2.13–2.17x.
The timing/numerical tables farther below describe the original port snapshot.

Run from the repository root, using the shared environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/clean-neurons/.venv/bin/python m16/approaches/sci_tao_v2/check.py \
  --checkpoint /workspace/clean-neurons/m16/approaches/mechanism_checks/checkpoint.pth
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/clean-neurons/.venv/bin/python m16/approaches/sci_tao_v2/benchmark.py \
  --checkpoint /workspace/clean-neurons/m16/approaches/mechanism_checks/checkpoint.pth
```

The first command writes `validation.json`. It checks 2,112 segments across
random DEV (H,T)=(1,2),(2,3),(4,3),(5,5),(8,6),(12,8),(16,10),(32,5), an all-zero
model, and two actual leaderboard DEV copies (first noise draw, seed=1, sigma
0.0025 and 0.02). The original checkpoint is used only to generate those DEV
copies (SHA256 `498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb`). Maximum old/new mass difference is **9.99e-15**, with identical cell
counts. Selected segments are independently checked on 65,536-bin forward grids;
maximum discrepancy is **3.91e-6**, within the conservative boundary-bin bound.
Zero-width windows and a deliberately exhausted cell limit pass.

| DEV model, 512 lines | Cells | Value-only prefix steps | Executed suffix steps | Old full-recompute steps | New charged FPE |
|---|---:|---:|---:|---:|---:|
| sigma 0.0025 | 4,213 | 1,366 | 17,971 | 42,130 | 4,765.02 |
| sigma 0.02 | 4,186 | 1,366 | 18,234 | 41,860 | 4,816.11 |

The legacy reported costs were 3,730.8/3,783.4: the new charge is **higher** than
that unjustified idealized charge, despite actually doing fewer dense products.
The old implementation's two-full-pass proxy is 8,426/8,372 FPE before separately
charging root and envelope work. This proxy is not an exact new-style FLOP count.
The Python reference walker is **11–44% slower in wall time** in the stored run on these batches, despite
lower arithmetic work, because caching/indexing adds Python and memory overhead.
Timing is host-sensitive; see the stored measurements rather than interpreting
this as a hardware speedup.

The second command writes `dev_benchmark.json`. It uses the unchanged estimator
on two DEV models, 1k/4k/16k requested budgets. A single independent 131,072-input
truth sample per model is a diagnostic, not leaderboard ground truth.

| DEV sigma | Requested budget | Actual FPE proxy | Estimate | Diagnostic truth (SE) |
|---|---:|---:|---:|---:|
| .0025 | 1,000 | 1,211.57 | .947095 | .941055 (.000651) |
| .0025 | 4,000 | 3,824.47 | .940397 | same |
| .0025 | 16,000 | 16,454.66 | .939954 | same |
| .02 | 1,000 | 1,185.69 | .575581 | .620193 (.001341) |
| .02 | 4,000 | 3,842.97 | .610706 | same |
| .02 | 16,000 | 16,237.92 | .620512 | same |

Two distinct supplied RNGs yield identical estimates/costs at each point.
The pilot uses estimated future costs, so requested budget can be overshot;
actual work is always returned. CPU time versus vectorized sampling at actual
cost is roughly 60–67x at 1k and 9.5–11.1x at 16k. These six estimates cannot
support a cross-model MSE or a challenge-solved claim.

A separate exact counterexample exposes a legacy tie-tolerance bug. With H=1,
T=3, Wi=1, Wh=0, Wo=[0,1e-14,0], face y=[0,0,1], moving pair (0,1), tau=.25,
the logits are constantly [0,1e-14,0]. The new wrong mass is .0987063257, the
closed-form `Phi(.25)-.5`; the old walker returns .1974126514 because it treats
a strictly positive constant logit gap as a tie. The new code tests actual
equality. This constructed case is separate from the 2,112 agreement cases.

### Optional compiled backend (bounded follow-up)

`walker.cpp` implements the same cached affine algorithm in a serial C++ loop,
with one line's state resident at a time. `compiled.py:walk_mass` preserves the
Python API, optional detailed counters, and failure behavior. Build explicitly:

```bash
/workspace/clean-neurons/.venv/bin/python m16/approaches/sci_tao_v2/build.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/clean-neurons/.venv/bin/python m16/approaches/sci_tao_v2/check_compiled.py \
  --checkpoint /workspace/clean-neurons/m16/approaches/mechanism_checks/checkpoint.pth
```

The compiler invocation is `g++ -std=c++17 -O3 -ffp-contract=off -fPIC -shared`.
There is no fast-math, native-CPU specialization, downloaded dependency, or
model-fitted parameter. The generated `_walker.so` is git-ignored. Compilation
is explicit, never triggered by import; selecting an absent backend raises with
build instructions. Set `SCI_TAO_V2_BACKEND=cpp` before importing the estimator to
use it; the default remains `python`. Standalone clients can directly import
`compiled.py:walk_mass`. A C++ compiler is the sole extra build requirement.

`compiled_validation.json` records the frozen checks: all 2,112 previous lines,
maximum Python/C++ mass difference **1.36e-14**, identical per-line cell counts,
all eleven counter arrays, and FPE costs. All-zero, tiny-gap, zero-width, and
cell-limit guards pass. The original independent dense-grid validation was not
repeated, since the compiled backend already agrees with that reference.

| DEV, 4,096 R23 lines | Legacy CPU seconds | Python cached | C++ cached | C++ speedup over legacy |
|---|---:|---:|---:|---:|
| sigma .0025 | .1495 | .1313 | .06585 | 2.27x |
| sigma .02 | .1554 | .1368 | .06732 | 2.31x |

These are medians of three warm CPU timings on the shared host. Speedup is
larger for small batches because the Python reference has substantial indexing
and dispatch overhead: the full frozen estimator is 11–12x faster at 1k,
8.8–9.2x at 4k, and approximately 5x at 16k. At 16k the C++ estimator takes
about .0305 seconds. All six DEV estimates and costs remain unchanged to
floating-point rounding. This check generated **no additional truth samples**
and ran no test-model accuracy experiments. Compilation plus validation finishes
in under 20 CPU seconds on this host, below the 300-second follow-up cap.

The compiled backend retains the same transparent conservative FLOP schedule,
including a 50-operation proxy for each Gaussian CDF. `0.5*erfc(-x/sqrt(2))`
replaces scipy `ndtr`; shape-based dense-product charges are identical. Scalar
branching can avoid some work charged to numpy's selects, so the schedule is
conservative for C++. Compiler instruction counts and libm internals are not
measured. Caching now takes O(T*H) scratch storage per serial batch, plus O(n)
output/counter arrays, rather than O(n*T*H) Python cache storage.

## Compute accounting

Counters are updated at the actual vector operations, per executed line. The
following conservative scalar-operation schedule includes comparisons/selects as
unit operations; memory copies, Python dispatch, and counter instrumentation are
not floating-point work. Consequently this is an **analytic FLOP proxy**, not
an instrumented machine-instruction count. Dense dot products have exact shape
counts; transcendental and branch overheads are explicitly estimated.

| Operation | Charge |
|---|---:|
| One prefix value step, finite guard | 2H² + 4H |
| One executed affine suffix step, right-hand gates and finite guards | 4H² + 20H |
| Gate roots for that step | 7H - 1 |
| Two readout products and finite guards per cell | 2T(2H - 1) + 4T |
| Crossed-prefix check / root minimum / cell bookkeeping | 9T + 12 |
| Each nonempty target half-interval | 20T + 12 + 4*CdfCost |
| Gaussian CDF proxy, CdfCost | 50 per call |

The current suffix recurrence schedule conservatively charges right-hand
gate selection, finite-state checks, and selects. C++ stops its prefix scan
early, but is conservatively charged the full per-cell bookkeeping schedule.
All four CDF calls made by numpy's eager `where` expression are charged, even
when the correct interval is empty. Scalar pilot and aggregate operations are
charged separately in the wrapper. FLOPs/FPE follow the shared v3 rule exactly:
`6000 * [T(2H²+2H)+2HT]/5760`. No model-dependent precomputation is hidden.
`validation.json` also records exact dense-dot shape FLOP counts for old/new.

Model-free precomputation: the inherited 160,001-point Gaussian face-CDF tables
have a declared 30M-operation rough proxy per T (including ndtr/exp proxies).
Reusable fixed Sobol point generation is also model-free and grows with the
requested face count; the single numeric metadata field cannot encode this
variable count. Neither cost is charged by the current shared rule. Tables,
interpolation, and the inherited tau<=8.5 clipping limit the overall quadrature
accuracy even though each resolved affine cell is integrated analytically.

The committed benchmark performs 18 estimator calls (six timed, twelve RNG
checks), totaling about 128,272 FPE, plus **262,144** diagnostic sampling forwards.
Validation performs 2,112 old/new line walks and 84 independent grids of 65,536
forwards each (5,505,024 diagnostic forwards); it is dominated by these small
validation grids. All runs use one CPU thread and finish in seconds, well below
the 600 CPU-second task bound. The checkpoint is reused; no training or large
fresh truth calculation is performed.

## Claims

- [measured] Cached prefix states and roots are actually reused; executed suffix
  counts are less than half the old full-recompute steps on the two trained DEV
  examples, with a one-time value-only prefix.
- [measured] Old/new random and DEV integrals agree to 1e-14; independent forward
  grids agree to their discretization tolerance. This is author validation, not
  independent confirmation.
- [measured] The tiny constant-gap example fixes one legacy tie error, and cap
  exhaustion now raises instead of silently returning an incomplete integral.
- [measured] There is no consistent wall-time speedup in the Python reference.
  The optional compiled backend is 2.27–2.31x faster than the legacy walker on
  the two predeclared 4,096-line DEV batches (three-run median CPU time).
- [inferred] Prefix caching is useful for more efficient compiled analytic line
  solvers; the present code provides a reusable correctness/accounting reference.
- [inferred] The 50-FLOP CDF approximation and conservative scalar schedules are
  adequate transparent proxies, but need DRI review before headline comparisons.
- [measured] The compiled backend gives about a 5x end-to-end speedup versus
  the Python reference for the six frozen DEV estimator checks at 16k; this does
  not by itself establish competitive MSE per wall-clock time.

## Alternatives not ruled out

The agreement test can share errors with the old geometry; forward grids test
individual lines, not the global co-area identity. The benchmark truth has
non-negligible sampling error. Only two trained DEV copies were checked, and
random weights are not representative of trained models. Lower arithmetic cost
need not improve MSE per wall-clock second. CDF costs may differ materially by
compiler/CPU; metadata does not measure them. No end-to-end leaderboard curves,
training-seed robustness, or new mechanistic interpretation were established.

## Failure modes

- Cached affine coefficients suffer floating-point cancellation on ill-conditioned
  models. This is numerical exact integration, not certified interval arithmetic.
- The inherited right-side probe `eps=1e-10` can skip extremely thin adjacent
  gate cells; it now invalidates crossed cached prefixes, preventing errors
  from persisting macroscopically. eps=0 uses explicit right-hand slopes. Exact-tie correction does not eliminate that limitation.
- More than `max_cells` per batch iteration raises; callers must not interpret
  it as a successful partial estimate.
- Heavy-tailed face integrands and adaptive pilot overshoot remain unchanged.
- For tiny models root/CDF work can dominate: honest cost is much larger than the
  legacy recurrence-only charge. Old small-model gains require fresh scoring.
- Arrays cache O(n*T*H) coefficients. Chunking limits peak memory, but very large
  H or T is not a promised practical performance regime.

## Follow-ups for the DRI

Register the new quadrature entry and review its FLOP convention. Port the
reusable walker to the other T20 estimators without changing their geometry.
Audit their near-tie tolerances and cell-limit truncation. Support an `all`
metadata declaration in the shared loader if arbitrary generic sizes are wanted.
Per-primitive profiling may refine the scalar cost schedule. The optional
compiled implementation below demonstrates a speedup without changing the rule.
