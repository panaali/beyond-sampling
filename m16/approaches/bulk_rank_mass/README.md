# T122: mass of exact rank-conditioned bulk certificates

## Question
How much probability mass do T120's 28 uniformly classified nonempty rank-conditioned regions cover on H4T4 seed0? Counts alone do not establish coverage.

## Method
Frozen before implementation, 2026-10-17; TASKS-only claim 7b89bb9. Reuse the same ten gate histories selected by gate mass in T120 (total mass .4482031037438533), their final exact feasibility classifications, and only their 130 existing exact T68 gate simplices. Initialize each simplex from exact cofactor facets and intersect it with all 12 ordered top/second rank cones (three coordinate-difference planes). Reuse the read-only exact integer clipper and positive GL20 mass primitive. No new neural gate enumeration or output splitting.

Controls: analytic signed orthants; exact parent-normalized-determinant conservation over all rank children; all children retain the parent orthant and satisfy rank planes; zero full-dimensional children for dual-certified empty pairs; independent existing correct-event simplex attribution excludes uniformly wrong cells and matches uniformly correct mass. Compare GL16 with unchanged GL20. Freeze source/artifact hashes. Cap 30 CPU seconds and 5000 clipping calls, with partial checkpoints after complete parents. Historical geometry construction (10.009618928 CPU seconds) is separately charged. A cap or implementation failure leaves unprocessed mass unresolved.

Two alternatives to useful coverage: uniformly classified cells may carry negligible mass (test by integration); geometry/mass errors may explain apparent coverage (test exact volume conservation, independent correct artifact and GL refinement). Even high coverage may be too expensive for an estimator (report complete incremental cost plus historical dependencies; no matched-cost claim).

## Status
Running; protocol frozen before code and measurements.

## Results
Pending bounded phase.

## Compute accounting
Charge source/data setup, exact clipping/triangulation and validation, quadrature calls at both orders, and correct-artifact controls. No random input samples. Exact witness computations and all setup count; source geometry cost remains a separate dependency. One thread, 2 GB cap, primary shared compute lock.

## Claims
[inferred] Exact feasibility certifies correctness or error on open rank-conditioned regions, but their mass must be separately computed.

## Alternatives not ruled out
Unselected histories, mixed rank cells, other models/seeds/noise, more efficient rank representations, and cheaper partial-mass estimators remain open. This is not a full-rung or challenge solution and does not assume errors concentrate near boundaries.

## Failure modes
Floating quadrature and input conversion retain their existing limitations; conditional bounds are not an unconditional rounding certificate. Shared source provenance reduces implementation independence. Strict rank/gate boundaries have Gaussian measure zero; identically zero gates were handled explicitly in T120. Exact topology does not make numerical mass exact.
