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
