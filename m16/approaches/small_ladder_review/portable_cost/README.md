# T52 delegated lane: portable cost of the T4 cone estimator

## Question
Does the successful H2T4/H4T4 accuracy calculation remain competitive when geometry, rule construction, determinants and positive cone integration are all charged transparently?

## Method
Claim pushed by root in96c79ad. This folder is the only editable lane. Import the DRI double-description (DD) source read-only and apply a local adapter, never change shared code. Freeze this protocol before implementation/runs.

Evaluate the two existing seed0 checkpoints only, using fixed GL orders4/8/12/16/20. Geometry is generated once but charged separately to every end-to-end curve point; quadrature setup is charged every time. Report numerical accuracy and analytic integration-only bounds, conditional on floating geometry. Use auditable scalar Gauss-Legendre Newton setup and pivoted4x4 determinant routines, independently checked against NumPy references. Correct identicallyzero logit margins with first-index tie ownership in the local outcome adapter. Matrix-rank/SVD remains an explicitly counted black-box kernel, not an invented exact FLOP number.

Portable cost is a symbolic arithmetic bound K + sum C_rank(m,n), with matrix-rank call shapes/counts exposed. Arithmetic/comparisons/bitset operations have unit cost; transcendentals have declared cost20, matching the DRI convention. Memory allocation, copies and Python dispatch are excluded from scalar arithmetic but included in measured CPU. Publish rank-kernel sensitivity separately, not as a certificate. No expensive new Monte Carlo truths: compare known independent references and exact tie controls. All numerical runs use the shared flock, single threads,2GB cap; each cycle <=120 CPU seconds.

## Status
Three bounded implementation cycles complete; independent accounting review pending. Root reviews this lane; Claude owns acceptance and shared-source changes. T60 tracks the discovered shared DD positive-volume tie bug.

## Results
Saved symbolic and bounded-rank cost curves are described below. No large-model or general noisy-ensemble success claim is made.

## Compute accounting
Include normalization, clipping predicates/reductions, adjacency bitsets, new rays, lifted maps, output/rank comparisons, triangulation incidence traversal, orthant splitting, L1 ray normalization, all determinants, quadrature nodes/weights, integration, and analytic bound calculation. Keep original DD Ops as a diagnostic only. Distinguish geometry reused for experimental convenience from geometry charged per estimator invocation. Calibration and reference/check costs reported separately.

## Claims
[inferred] Existing DD Ops does not include the dominant positive-volume integration, root review normalization/determinants, or all combinatorial bookkeeping. [inferred] Positive-volume duplicate readouts require a local correction before generic correctness tests. No cost superiority or formal end-to-end error certificate is claimed yet.

## Alternatives not ruled out
1. A favorable ratio may come from omitted quadrature/setup. Falsifier: doubling n should increase node arithmetic approximately eightfold while geometry stays fixed; cold setup must have nonzero cost.
2. A favorable estimator may silently duplicate/drop tied regions. Falsifier: allzero and duplicate readouts must yield1/4 using the counted geometry path, without skipping the computation as a free analytic shortcut.
3. Black-box rank computations may dominate. Falsifier: expose every call shape and sweep explicit kernel prices over a wide range rather than assuming their existing proxy is exact.
4. Replacing setup arithmetic could alter answers. Falsifier: polynomial-exactness checks for Gaussian rules, determinant comparison, and unchanged trained-checkpoint probabilities.

## Failure modes
DD floating tolerances and finite bitset capacity; zero or degenerate input weights outside tested base scope; extreme skinny cones; SVD runtime is not a fixed arithmetic count; CPU results depend on hardware and batching. The quadrature bound excludes geometric and roundoff error. Numerical verification alone does not certify all generic weights.

### Cycle 1 implementation checks
`check.py` completed in0.984 CPU seconds including torch import/checkpoint loading. Gaussian rules n4/8/12/16/20 integrate every monomial through degree2n-1 within2.23e-16. Twenty ordinary/nearly dependent4x4 determinant cases agree with NumPy within2.67e-15 absolute error. Counted DD plus local first-index tie handling gives0.2499999999990603 for both allzero and identical readouts, within the analytic integration bound of1/4. The DD source is unchanged. `checks.json` saves exact costs and cases.

### Cycle 2: fixed precision versus symbolic operation cost
The two-checkpoint curve took3.182 CPU seconds total. DD gives258 correct simplices for H2 and5040 for H4. Explicit GL20 results are0.30773074302392717 and0.47701312599411877, agreeing with independent geometry references. Full positive integration dominates the old geometry-only counters: known arithmetic upper bounds at GL16 are47,498,549 (H2) and932,535,166 (H4), plus293 and6797 matrix-rank kernels respectively. These are symbolic totals, not complete numeric FPEs.

The curve exposes four rank-price scenarios, including zero (clearly labeled an incomplete lower bound) and an intentionally extreme one billion operations per rank call. Even under the latter assumed price, GL16's integration-bound squared remains below equal-operation sampling variance on both examples. This is sensitivity evidence, not a proof of the SVD implementation's cost, and the integration bound still excludes floating geometry. `results.json` contains all orders4/8/12/16/20, component counters, assumptions, CPU, and bounds. No new model/noise sweep was run. Two65536-input sampling batches are timing-only; their outcomes do not enter the estimate or reference.

Next bounded cycle: replace the remaining small-rank black box with an explicitly bounded scalar rank routine, fail closed on ambiguous threshold decisions, and compare every decision/geometry result with SVD on the two existing examples. Keep numerical correctness checks separate from charged production work.

### Cycle 3 frozen bounded-rank protocol
Replace only the matrix-rank predicate with row-pivoted, twice-reorthogonalized Gram-Schmidt, capped at min(m,n) basis vectors. Verify a numerical singular-value bracket: residual Frobenius norm bounds the first omitted singular value; a selected coefficient minor's inverse-residual/Frobenius bound supplies a lower bound for retained singular values. If this bracket does not straddle the original1e-9 threshold with a conservative floating cushion, raise an explicit ambiguity error. This is a bounded arithmetic algorithm, not interval-certified floating linear algebra.

Falsifiers before runs: compare every rank decision with the original SVD on both existing checkpoints, and test zero/rank-deficient/well-separated/threshold-near matrices. Near-threshold ambiguity must be reported, not forced into a rank. Expose validation SVD calls separately; production cost must contain zero opaque numerical kernels. No new noisy-model sweep. If decisions disagree or geometry probabilities change materially, retain the SVD-symbolic result and report failure.

### Cycle 3 results: bounded arithmetic rank
`check_rank.py` passed all ten fixed fixtures, including zero, exact rank deficiency, four random factorizations with seed52031, and diagonal singular values below/above/at the1e-9 threshold. The exact-threshold case correctly raises `AmbiguousRank`. Fixture matrices are saved verbatim. Every rank call agrees with the original SVD:293 for H2 and6797 for H4. Correct simplex counts remain258 and5040, and GL16 probabilities are bit-for-bit identical to the earlier saved values. H2/H4 checks took0.934/3.054 CPU seconds including checkpoint loading and separately counted validation SVDs. These timings are validation costs, not a claimed production speedup.

Production arithmetic now has no opaque numerical kernels. Full GL16 known-operation upper bounds are47,814,526 for H2 and939,471,674 for H4, including geometry, bounded rank, determinant and Gaussian-rule setup. The integration-only bounds are1.168e-6 and1.769e-6 respectively. Their squared-bound/equal-operation sampling-variance ratios are4.78e-6 and6.13e-5 under the declared cost convention. GL20 totals are91,246,470 and1,785,667,522, with bounds1.786e-8 and2.705e-8. `rank_curves.py` substitutes the checked geometry cost into the existing saved curves; it performs no new model or quadrature evaluations. Sampling receives free RNG, target ranking and argmax in this conservative arithmetic comparison.

The numerical-rank bracket is justified in exact arithmetic as follows. Write A=CQ+E, with Q formed by twice-reorthogonalized row Gram-Schmidt. The residual Frobenius norm bounds the first omitted singular value. A selected square row minor T of C gives sigma_min(C)>=sigma_min(T). For an approximate inverse X, sigma_min(T)>=(1-||I-TX||_F)/||X||_F when the numerator is positive. Also sigma_min(Q)>=sqrt(1-||QQ^T-I||_F). Combining these and subtracting ||E||_F bounds the smallest retained singular value. The code adds a declared heuristic1e-12*max(1,||A||_F) cushion and fails closed if the rank threshold is unresolved. This cushion is not an interval-arithmetic roundoff proof. The production loop has at most min(m,n) basis steps and scalar inversion dimension at most4; all matrix multiply/add/norm operations are explicitly charged.

Remaining limits: the cost ledger is a source-audited scalar arithmetic upper bound, not a machine-instruction or memory-work bound; independent audit is pending. The quadrature error excludes floating geometry and roundoff. Original DD SVD results remain preserved as a distinct fallback reference, but the bounded production method never silently invokes that fallback. These two nominal checkpoint curves alone do not establish the ARC challenge criterion across independent noisy copies or the H16,T10 model.

Reproduction (run from this worktree, with the resource cap and single-thread environment):
```
ulimit -v 2500000
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python m16/approaches/small_ladder_review/portable_cost/check_rank.py --H 2
flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python m16/approaches/small_ladder_review/portable_cost/check_rank.py --H 4
/workspace/clean-neurons/.venv/bin/python m16/approaches/small_ladder_review/portable_cost/rank_curves.py
```
