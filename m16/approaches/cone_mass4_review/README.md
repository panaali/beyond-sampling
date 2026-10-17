# T119 independent T109 trained-cone mass verification

## Question
Does the author's new single-edge Plackett primitive repair the measured trained T4 mass failures, and what does it cost after all setup and failures are included? Claim e4c9c77 was pushed before implementation. This is independent verification; the T109 author owns changes to that algorithm.

## Method and frozen protocol
Pin author mass4.py at published commit7b7279a, SHA6617ddd0a042d4af5fbc899e0542b9e31c1ee23f5ab8d47e5fb8ee2c005f05a6, as an unchanged local snapshot. Its adaptive node choice and signed flat-cone decomposition remain unchanged. Do not import the author's reference as independent evidence; use separately written analytic/positive checks.

Phase1: independent block-diagonal bivariate orthant products. For rho in{−.9,0,.8,1−1e−6} and second rho in{−.7,0,.6}, construct defining normals by analytic2×2 Cholesky blocks. Exact probability is[1/4+asin(rho)/(2pi)] times the second factor. Test normals and rays interfaces, positive row/ray scaling by powers2±20 and fixed coordinate permutation. Include coordinate orthant rotations/permutations, zero determinant rejection/reporting and nearly flat but invertible synthetic rays; retain all exceptions and negative values. Exact control target1e−7, no fitted thresholds.

Phase2: replay all fixed captured quadratic6347 and nearest11421 trained H4T4 error simplices, hash-pinned to T110. Recompute every mass, preserve failed/negative values, cold GL-rule cache/setup and author counter separately; full CPU includes source/module initialization plus rule setup. Per corpus<=100CPU,2GB, one thread/primary lock. No geometry enumeration or model/parameter sweep. Compare summed error with independent conditional reference.477013125994119, not with default biased masses. Per-family corrected error sums follow cumulative recorded simplex counts; no changed output attribution.

Phase3: predeclare independent checks on16 captured quadratic rays: four evenly spaced indices0,2115,4231,6346; four smallest absolute determinant; four most negative old default masses; four largest old default masses (deduplicate deterministically, no backfill). Reference uses positive simplex Gaussian integral, with analytic conditioning/sector splits if needed, or a separately implemented high-precision all-correlation path. Freeze exact implementation and numerical bounds before execution; <=100CPU cycle and explicit incomplete references. Same-ray agreement between old and new formulas alone is insufficient. If controls fail, localize without editing author snapshot; source-level independent derivative/degenerate-endpoint checks precede any author repair request.

## Alternatives and falsifiers
1. Speed could hide inaccurate signed cancellation or float singular endpoints: analytic product controls, scale/permutation, and positive/high-precision references test this.
2. Good per-simplex error may accumulate into inaccurate network probability: full corpus aggregate compared to the independent geometry reference, every negative/failure retained.
3. Cheap author counters may omit GL initialization, inverse/setup, recursion and failed calls: instrument cold source/cache/wallCPU and clearly separate partial named conventions; no portable total-cost claim without an audit.
4. Disagreement may originate from trained floating geometry, not mass formula; all rays unchanged and model/reference provenance pinned. Preserve unknown reference error rather than selecting whichever value looks favorable.

## Status and results
Protocol frozen; implementation pending. No success, exactness, sampling or solved-rung claim.

## Compute accounting
Use primary lock for<=100CPU phases, review lock only<=20CPU controls;2GB/one thread. Fixed source/array reads and diagnostic references separately charged. Author elementary-operation count is an object of review, not accepted complete cost. No random model inputs or Monte Carlo.

## Claims
[inferred] Same-ray replay isolates a mass primitive's effect from gate geometry.
[speculative] T109 meets useful total accuracy and cost on the trained corpus.

## Alternatives not ruled out and failure modes
Signed cancellation, singular endpoint classification, roundoff at nearly dependent rays, insufficient adaptive order, accumulated per-cone error and incomplete cold setup accounting. A failing finite panel does not retire single-edge integration or signed decomposition generally. Author implementation remains untouched; DRI decides acceptance.
