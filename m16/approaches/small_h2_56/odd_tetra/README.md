# T71: three analytic simplex coordinates for T5

## Question
Can the first spherical moment on S3 remove three T5 simplex coordinates and improve complete cold accuracy estimation?

## Method
Root pushed claim b7209a1 before implementation. For four endpoint columns V in R5, thin QR gives V=QR and c=R^-T1. Radial projection gives dOmega3=|detR|*||Ry||^-4 dy and c dot omega=1/||Ry||. Thus the standard tetrahedron integral of ||Vy||^-5 equals ||c||/|detR| times its first spherical moment in direction c. On S3, Delta(u0)=-3u0, so that moment is minus one third the sum of outward conormal components times S2 facet areas. Each facet area has an elementary positive atan2 formula. T55 independently checked the identity, sign convention, Jacobian and lack of an extra factorial before implementation.

Prototype uses oriented face cofactors and positive solid angles, guarding rank and boundary cancellation. A positive original-domain fallback uses matched Jacobi quadrature and the previously checked I5 segment formula, with the independently reviewed generic positive Chebyshev tail. Analytic rounding and geometry remain uncertified. T5 only; no T7/T9 work until demonstrated.

## Frozen initial protocol and alternatives
First test identity endpoints (truth2*pi/3), equal endpoints (norm^-5/6), collinear endpoints, deterministic signed-L1 endpoints, thin/asymmetric endpoints, orientation permutations and positive scaling. Compare ordinary cases to positive original-domain quadrature; preserve failed cases and count every analytic attempt/fallback. Before trained runs, verify the one-outer-coordinate Jacobian with constant integrands and the identity-ray Gaussian orthant1/32. Then compare the full T5 base and existing fixed .0025/.02 noisy references; no new Monte Carlo. Charge cold geometry, planning, primitive setup, elementary functions, fallbacks and outer integration.

Alternative1: a sign/factor error can reproduce selected shapes accidentally; exact identity/equal controls and permutation invariance can falsify it. Alternative2: facet cancellation may erase the expected savings on thin trained cones; measure all rejection/fallback counts and compare against positive quadrature. Alternative3: per-call QR/cofactor setup may dominate despite fewer coordinates; record full cold CPU and scalar component costs, without equating wall speed with audited FPE.

## Status
Initial controls, trained pilot and full three-model comparison completed. Independent asymmetric-thin review is pending. No solved or equal-compute claim.

## Results
Full comparison in `full.json`, with costs and limitations below.

## Compute accounting
Every numerical cycle acquires /workspace/work/codex-compute.lock, uses one thread and ulimit -v2500000, at most110CPU seconds, and node blocks<=65536. Explicitly authorized review work may use the separate review lock for<=20CPU seconds. Peak memory<=2GB. Initial controls use the compute lock. Total FPE remains unaudited until geometry/library/setup costs are counted.

## Claims
[inferred] The spherical divergence identity removes three coordinates in exact arithmetic. [speculative] A stable implementation offers practical savings on all three trained T5 cases.

## Alternatives not ruled out
Conditioning, boundary cancellation, fallback-dominated cost and floating geometry errors.

## Failure modes
Rank-deficient tetrahedra and nearly cancelling facet sums require positive fallback. Tight quadrature bounds do not certify floating arithmetic. Shared signed-L1 planes are used only after checking their residual; incorrect normals cannot silently enter the formula.

## Initial primitive result
`controls.py` passes15 fixed identity/equal/collinear/deterministic signed and thin tetrahedra, two orientation permutations and positive scales1e-3/1e3. Maximum relative difference from independent positive original-domain I5/GL96 quadrature is4.32e-14; identity truth2*pi/3 and equal-endpoint truth||A||^-5/6 pass. The full control cycle uses0.067CPU seconds. `controls.json` preserves every analytic/fallback diagnostic. This supports normalization and numerical behavior of the primitive, not trained-model performance or certified floating arithmetic.

`primitive.py` forms signed facet cofactors in a thin-QR frame, projects a verified known plane normal when available, and computes positive S2 facet solid angles. Small singular-value ratios, excessive estimated boundary cancellation or violated positive geometric bounds trigger original-domain Jacobi+I5 quadrature. Its error bound is one third of the uniform Chebyshev tail (two positive remainder errors times tetrahedron volume1/6). QR/SVD/cofactor determinants, atan2 calls, rejected attempts and all fallback nodes are counted separately; complete scalar-operation costs remain to audit.

## Frozen trained pilot
Use32 equally spaced indices from each existing T5 base, noise.0025 and noise.02 correct-cone cache. Outer Jacobi alpha3, beta0 at fixed orders8 and16; four inner endpoints are u*v0+(1-u)*v_i, i1..4. Check weights sum1/4, multiplied by tetrahedron volume1/6 gives1/24. Identity5-ray cone must approach1/32. Compare unchanged full-cone Jacobi/I5 integrals at the same orders with inherited geometry-only bounds; report all differences and fallback costs. Only after this pilot, rebuild cold full base/noisy cases with their previously frozen1e-4 per-model bound policy. No new random validation or data-selected orders.

## Trained pilot result
The one-coordinate tetra method agrees with the old three-coordinate I5 rule at order16 to at most1.42e-16 on all three fixed32-cone trained/noisy subsets. The identity-ray orthant agrees within the expected outer error, and the1/4 outer weight times1/6 tetra volume confirms normalization. However,25–36% of tetra calls trigger positive fallback, so the prototype's universal1e-11 inner tolerance creates substantial avoidable work. `pilot.json` preserves these negative cost diagnostics; dimensional reduction alone does not guarantee speed.

## Frozen full comparison and inner error allocation
Before full trained runs, replace the unnecessarily strict universal tolerance by a fixed total positive-inner budget1e-7. Each cone receives1e-7/N; its uniform tetra absolute tolerance is that amount divided by C5*|detV|*sum(outer_weights). Positive weights guarantee the total propagated inner error<=1e-7. Root previously checked this same allocation algebra for T6. Retain the prior frozen geometry-only1e-4 outer-bound policy (Jacobi, centroid radial lower bound, finite order list, equal per-cone allocation), so old and new integration estimates are directly comparable. Rebuild all three model geometries cold, preserve every failure/cost and compare to their existing saved references. No pruning, new Monte Carlo or reference-derived parameters. Use the same per-cone orders in both tetra and I5 cost comparisons; do not select models or orders by outcome.

## Full trained comparison
[measured] All three cold geometry rebuilds finish. Estimates for base, relative noise .0025 and .02 are respectively .27923737131493037, .31063601432908156 and .32506108240958753. Differences from the saved independent/refined references are at most 4.104e-11; differences from the old I5 rule at identical outer orders are at most 5.585e-11. Base full gate mass is .9999999999977828. These are validation differences, not certified arithmetic errors.

| Noise | Outer quadrature bound | Positive inner bound | Integration CPU | Cold CPU including checkpoint load | Tetra calls | Positive fallbacks | Fallback I5 nodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2.19040e-5 | 2.03074e-9 | .689 | 2.741 | 44,514 | 7,565 | 289,888 |
| .0025 | 2.14788e-5 | 9.41838e-10 | 1.182 | 3.959 | 87,762 | 12,568 | 285,936 |
| .02 | 2.09620e-5 | 1.12957e-9 | 1.265 | 4.365 | 88,146 | 14,423 | 359,680 |

Cold CPU includes geometry, bound planning and integration, charging each case the same measured .840 CPU checkpoint loading. Validation work is separate. Geometry's recorded omitted-mass upper bounds are below 3.84e-14, conditional on its floating classifications. Compared with the pilot, allocating the inner tolerance by positive contribution substantially reduces fallback work. Nevertheless, 14–17% of calls still fall back. The old I5 rule uses 7.13/15.82/14.18 million nodes at the same orders; tetra calls are substantially more expensive individually.

Every tetra call includes QR and SVD; attempted analytic calls additionally include a 4x4 determinant, sixteen 3x3 cofactor determinants, plane projection and four atan2 calls. `full.json` retains all counts, including rejected attempts, fallback bound trials and fallback segment nodes. These library operations and geometry setup have not been converted to a complete scalar-operation upper bound, so total FPE remains infinite/unaudited. The heuristic cancellation test is not an interval arithmetic certificate, and the quadrature bounds do not cover geometry or analytic rounding.

Reproduce from the repository root:
```sh
ulimit -v 2500000
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-compute.lock /workspace/clean-neurons/.venv/bin/python m16/approaches/small_h2_56/odd_tetra/full.py
```

## Independent adversarial review
T55 review f07fc00 uses asymmetric coalescence toward p=(.8,.05,.05,.05,.05), where the perpendicular foot lies outside the tetrahedron and signed boundary terms cancel. Independent 70-digit positive original-domain GL12 agrees within 8.95e-13 on accepted epsilon .01/.001 cases; all five smaller cases trigger positive fallback and agree within 3.42e-16. The reviewer checked signs, Jacobians and the fallback factor. This strengthens numerical evidence but does not certify floating error.

## Frozen fixed-tangent optimization
The outer endpoints are u*v0+(1-u)*vi, i=1..4. Their three edge directions span a fixed subspace for each cone. Compute one thin QR and condition diagnostic per cone; express all endpoints in this common tangent basis and a perpendicular coordinate delta. Same-orthant signed-L1 normalization implies delta>=1/sqrt(5). Face cross products and the edge determinant can be computed once per cone, replacing repeated QR and sixteen cofactor determinants. Retain the old implementation and positive fallback unchanged for validation.

Alternative1: the fixed-frame sign or powers of (1-u) can be wrong despite regular-shape agreement; compare against the old primitive on identity, reflected/permuted and deterministic trained pilot cases at both frozen orders8/16. Alternative2: removing per-node SVD can miss thin configurations; use an explicit conservative lower bound on singular ratio based on the block inverse Frobenius norm, and compare fallback classifications and asymmetric coalescence. No trained full runs until these controls pass. Outer orders and allocated inner budget remain unchanged. Count setup operations separately and retain all rejected attempts; no complete-FPE claim until the geometry/library audit is done.

## Fixed-tangent controls and full comparison
[measured] All48 fixed-frame controls pass, including the prior32-cone trained subsets, identity, reflected permutations, asymmetric coalescence epsilon1e-2 through1e-8 and equal endpoints. Maximum relative difference from the old primitive is1.63e-11; all thin cases use the positive fallback. T55 independently checked the face orientation, delta and (1-u) powers, and the inverse-Frobenius singular-ratio lower bound. Control CPU .371 seconds.

The equal-endpoint control exposed a pre-existing floating-domain bug: the computed radial lower bound could exceed the upper bound by one ulp. `fixed_controls_failure.json/.npz` retain the failed case. The radial helper now enforces their mathematical ordering by capping the lower bound at the upper bound. This repairs the domain and is not a floating-error certificate. All original15 controls pass unchanged after the repair.

[measured] `fixed_full.json` repeats the same frozen three cold model comparisons. New estimates agree with the original tetra implementation within5.56e-17; the base partition closes within2.22e-12. Integration CPU is .468/.809/.941, versus .689/1.182/1.265 for repeated frames. Cold CPU including loading is2.463/3.674/4.055. Each cone now uses one QR/SVD (5867/10970/11486 total), with no per-node determinant or linear solve; seven cross products per cone replace the cofactor construction. The conservative condition screen adds2/7/11 positive fallbacks. Complete FPE remains unaudited.

A diagnostic maximum over all nodes initially includes enormous cancellation ratios for already rank-rejected nodes; none is accepted. The implementation now reports eligible-node, accepted-node and all-node maxima separately. This reporting correction does not change decisions or estimates.

## Frozen counted composition
Compose the unchanged T52 portable T5 geometry with a fixed24-point Gauss-Legendre outer rule and weight (1-u)^3. Use its already reviewed Bernstein outer bound at degree22; reserve a total positive inner budget1e-9, allocated equally per cone and propagated with positive weights. No order selection from references. Replace opaque QR/SVD with two-pass modified Gram-Schmidt, checking its orthogonality and the existing explicit inverse-Frobenius condition lower bound; replace fallback Jacobi rules with counted Gauss-Legendre rules, degree n-2 and the same positive Chebyshev tail. Preserve original and fixed-frame engines for comparison. Count every setup/bound attempt, failed analytic attempt and fallback node, using the existing T52 convention of20 arithmetic units per elementary transcendental. Bound-specific rising factorials use finite products rather than an opaque gamma function. Verify fixed controls before cold base/noisy composition, record all failures, and retain geometry/roundoff caveats even if no opaque kernels remain.

Alternative1: counted Gram-Schmidt may change classification or lose orthogonality on thin cones; reject orthogonality failures and compare all existing adversarial controls with the saved independent engine. Alternative2: cost savings disappear under positive fallback or geometry setup; report full component costs and matched conditional-error versus sampling, without treating reduced numerical dimension as proof of efficiency.

## Counted composition result and retained failure
The first composition with T52 DD geometry stopped on all three models because its orthant clipping leaves wrong-sign coordinates of order4e-17; `counted_failure.json` retains the failures and their partial costs. No ray was clipped or projected to suppress them. The fixed-frame identity only requires a shared affine plane separated from zero, so validation now checks the signed-normal plane directly. Maximum residuals on the three models are4.44e-16/5.55e-16/4.44e-16. The Cauchy radial helper also clamps negative minimum projections to zero before squaring, as required for general supporting planes. Floating geometry remains conditional.

[measured] Counted cold DD+tetra composition completes all three fixed models with no opaque numerical kernels. Fixed GL24 estimates are .27923737131514975, .3106360143298103 and .3250610824117577. Differences from T52's independent DD+I5 GL24 values are below2e-15. Outer quadrature bounds are6.524e-7/6.972e-7/7.260e-7 and separately propagated positive inner bounds2.038e-11/1.313e-11/1.840e-11. All24 counted controls agree with the original engine within1.51e-11 relative; finite-product Chebyshev tails agree with the independently checked generic bound within1e-13.

| Noise | Proposed full scalar bound | Forward equivalents at65 ops | Cold CPU, provided model | Conditional squared-bound / sampling variance | Reference-interval ceiling squared / sampling variance |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 |204,838,786|3,151,366|1.519|6.67e-6|1.40e-10|
| .0025 |308,432,180|4,745,110|2.819|1.08e-5|.0121|
| .02 |370,263,176|5,696,357|3.052|1.37e-5|.0132|

The full scalar bounds include geometry, rules, bounds, rejected analytic attempts and all fallback nodes. They are proposed operation upper bounds pending independent review, not certificates for rounding or geometry. The T52 forward denominator65 counts only affine operations and therefore favors sampling. Model input preparation/checkpoint loading (.885 CPU shared) is separate; the usable `estimator.calculate` additionally charges128 operations for provided-weight validation. Compared with the earlier T52 GL24 count, these proposed bounds are about20/24/21 times smaller. The comparison uses the entire cold estimator and the same conditional quadrature policy, not cached model geometry.

`estimator.est(model,budget,rng)` exposes the required interface; its fixed24 rule treats the budget as advisory and never uses rng or reference estimates. Until independent cost review the adapter conservatively returns infinite cost, while `calculate` exposes the complete proposed inventory. `COUNTING.md` gives the explicit line-group proof obligations. The worst reference-interval MSE ratio is below .014, conditional on those reference intervals and floating geometry; this is not a completed challenge claim.

Reproduce with the same thread/memory/lock prefix as above and script `odd_tetra/counted_full.py`. That script preserves all three cases, geometry diagnostics, source hashes, counters and failures. Validation CPU8.274 seconds, peak RSS221MB.

## Interface check
Package import and provided-model `calculate` complete on the base checkpoint with the unchanged estimate .27923737131514975 and full proposed count204,838,914, including128 operations for weight validation. `interface_check.json` records1.467 CPU. The thin adapter returns infinite cost while independent review is pending, and does not inspect rng; budget remains advisory. Relative imports avoid collisions with other agents' modules named `volume` or `segment`.

## Frozen counted partition check
Validate the completed counted MGS/GL pipeline on the full gate partition of the same base and two fixed noisy models, using the unchanged24-point outer rule and1e-9 positive-inner budget. These are validation-only geometry enumerations, with their own counts/time; no accuracy estimates or order policies are changed. Require mass one within each reported conditional quadrature bound, and retain numerical differences even if much smaller. This checks the new general-affine-plane input adapter and MGS/positive-fallback composition independently of the target event.

## Counted partition result
[measured] `counted_partition.json` reports full gate masses1.000000000000005,1.0000000000269944 and .9999999999994292 for base/.0025/.02. Absolute discrepancies are5.11e-15/2.70e-11/5.71e-13, all below their conditional outer-plus-positive-inner bounds2.3822e-6/2.3807e-6/2.3579e-6. These independent target-free checks preserve all gate pieces and use the unchanged counted primitive. Validation costs are recorded separately from estimator costs.
