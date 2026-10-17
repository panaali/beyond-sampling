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

## First independent results and frozen failure localization
All72 analytic product/interface/scale/permutation cases and3 independently positive skinny-cone controls pass the frozen1e-7 absolute check. The cold quadratic corpus takes.260CPU and3,288,370 author operations, estimate.477013704055 (+5.78e-7 from reference); however index5826 returns−3.6109944e-7. Any true Gaussian mass is nonnegative, so this alone disproves the1e-7 per-cone target on that input. Nearest completes all11421 attempted calls in.531CPU/5,746,100 author operations, but index420 raises `ValueError: math domain error`; no aggregate accuracy is reported. Indices495/10506 return−6.15e-9/−2.02e-7. Preserve every exception, negative and cold-cache event. This is a numerical implementation/precision failure on a bounded panel, not a failure of the single-edge method.

Freeze failure localization before changes: reproduce these four named problematic cones, save defining rays/hashes and selected numerical exception frame scalars. Test explicit author n=20,40,80,160 on the negative cones only, preserving default and every result; no production change or tuning claim. Node stability cannot certify correctness. For domain failure, inspect the actual failed asin/sqrt argument and determine whether a narrowly scoped roundoff guard is justified before proposing any patch. Author source remains unchanged. Independent high-precision all-correlation reference will include the originally frozen subset plus these four diagnostics, with bounded progress and failures saved.

### Frozen independent reference implementation
The16-index quadratic subset is fixed by the protocol;12 share a signed coordinate orthant and four do not. For same-orthant rays, reflect signs, normalize each ray to L1=1, then use positive20-point tensor Legendre integration of determinant/[2pi²||sum y_i r_i||⁴] over the barycentric tetrahedron. The independent integration-only bound is det*(12n+9)/(6pi²*3^(n−1)), conditional on exact geometry/arithmetic, as previously independently checked in T51/T55. This avoids signed Plackett cancellation and gives positive values. Include the three same-orthant failure cones5826/420/10506 with this reference; nearest495 has a mixed tiny third coordinate and gets the high-precision path.

For mixed cones, independently implement the all-correlation path directly with2×2 Schur-complement formulas, not the author's reference/inverse-at-each-node code. Construct facet normals from input float rays at50 and70 decimal digits, integrate with tanh-sinh on fixed panels0,.5,.9,.99,.9999,1; record both values and callback/time counts. Agreement is not a rigorous quadrature certificate. Bound the whole phase at100CPU and save partial progress after every case; preserve any unresolved reference rather than treating it as author failure. No reference informs node/threshold choice.
