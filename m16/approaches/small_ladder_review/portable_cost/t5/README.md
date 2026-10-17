# T52 counted H2T5 extension

## Question
Can the independently checked T5 cone calculation compete with sampling when geometry, rank, determinants, quadrature construction, analytic segment evaluation and error-bound planning are all charged?

## Method
Parent pushed the T52 claim extension in18fdee3. Edit only this subfolder and preserve T4. Reuse the source-audited DD wrappers and first-index outcome adapter, with a width5 bounded-rank routine and five-dimensional orthant triangulation. Use the exact positive I5 segment formula to eliminate the last Duffy coordinate, leaving three ordinary Gauss-Legendre coordinates. This is deductive integration, with no random model inputs.

For L1-normalized same-orthant rays R, mass is3|det R|/(8*pi^2) times the four-simplex integral of||Ry||^-5. The positive segment formula is independently checked in T52/t5_independent. Ordinary GL order n has valid Chebyshev approximation degree K=n-2: a degree-K polynomial in q=||Ry||^2 has coordinate degree at most2K, and the outer Duffy Jacobian has maximum degree3. Thus2K+3<=2n-1. Exact inner integration preserves this polynomial exactness. Use the existing Bernstein-ellipse bound, with every trial charged; geometry and floating error remain separate limitations.

## Status
Protocol frozen before implementation. First verify primitive degree/cost and base geometry; then evaluate only seed0 and the two existing T48 relative-noise copies sigma0.0025/0.02. Their RNG resets to seed1 for each sigma, with the precise multiplication order from models.noisy. Save source/model/checkpoint hashes and every ambiguity/failure. References are the T52 independent DD base estimate and T48 saved noisy estimates; no new MC.

## Results
Pending. Initial fixed rules16/24/32, charged cold setup and full geometry separately at each point. Stop each CPU cycle before120 seconds; serialize through the shared compute lock, one BLAS thread,2GB cap, node chunks<=65536. Primitive checks below20CPU may use the review lock.

## Compute accounting
Scalar arithmetic/comparisons/bit operations cost1; transcendental functions cost20. Reuse T4 conventions for memory/interpreter exclusions. Explicitly charge determinant5x5, bounded rank with width5 control allowance, setup, Bernstein bounds and all segment dot products/square roots/arithmetic. Expose reference-only SVD calls separately if used. Affine-only sampling denominator is T*(2H^2+3H-1)=65 operations for H2T5; RNG, ReLU, ranking and argmax are free to sampling in this conservative comparison.

## Claims
[inferred] Analytic inner integration should reduce quadrature work from n^4 to n^3 without changing the inherited degree argument. [measured] None yet for the counted T5 implementation. No self-confirmation or total floating-error certificate.

## Alternatives not ruled out
1. Wrong Jacobian/degree could make the bound optimistic. Falsifier: exact simplex polynomial moments through the predicted boundary degree, plus identity-orthant mass1/32.
2. Width4 assumptions could corrupt width5 geometry or costs. Falsifier: base DD cell/simplex counts and all rank decisions agree with the independent reference; scalar5x5 determinants agree with independent NumPy checks.
3. A favorable comparison could omit segment/setup/special-function costs. Falsifier: save component ledgers and cubic node scaling with cold rule construction; no free rank or bound-planning kernels.
4. Noise could change topology or trigger rank ambiguity. Falsifier: retain both pre-existing noisy cases and all failures, comparing to frozen independent geometry references.

## Failure modes
Numerical DD tolerances, rank ambiguity, ill-conditioned rays, finite bitset capacity and polynomial-bound looseness. The quadrature bound does not certify floating topology, roundoff or the complete challenge at H16T10. No hard budget guarantee: fixed rules may exceed an advisory budget and must report their full charged work.

### Cycle 1 primitive and geometry checks
`python -m m16.approaches.small_ladder_review.portable_cost.t5.check --geometry` completed in1.729 CPU seconds under the review lock. All126 simplex barycentric moments through degree4 at GL4 agree within2.78e-17. The18 saved independent100-digit segment controls agree within7.24e-16 relative error. Twenty scalar5x5 determinants pass the NumPy comparison; width5 rank fixtures pass, including explicit near-threshold ambiguity. Identity-orthant estimates satisfy their analytic bounds around1/32.

Base H2T5 DD gives110 gate cells and2039 correct simplices, exactly matching the independent review. Every one of3076 bounded-rank decisions agrees with the validation-only SVD. Production geometry has no opaque numerical kernels and costs28,911,967 declared scalar operations. The validation cycle records SVD calls separately and is not a production timing result. Full integration/cost curves are next.

### Cycle 2 results and retained failure
The fixed-rule curve completed base and sigma0.0025, then the sigma0.02 case explicitly failed its bounded-rank bracket. Total CPU31.921 seconds; every completed result and the failure's partial geometry cost are retained in `results.json`. Base GL24/32 gives0.2792373713151490/0.2792373713151495, agreeing with independent DD reference within2.5e-15. Sigma0.0025 gives0.3106360143298084/0.3106360143298089, within the earlier independent Qhull integration bound.

GL16's conservative bound is insufficient for the equal-arithmetic sampling comparison: squared-bound/variance ratios29.3 and58.2. This does not demonstrate inaccurate estimates; it means this bound cannot prove efficiency at that order. GL24 ratios are1.31e-4 and2.56e-4; GL32 ratios2.55e-10 and4.89e-10. Full operation costs, cold setup and bounds are saved per point. The public fixedGL24 adapter is supplied but remains unsupported on the retained sigma0.02 ambiguity until that gap is addressed.

The failure reports retained rank4 lower bound1.4805e-10, omitted bound1.9991e-12, and threshold1e-9. It could be a genuinely sub-threshold singular direction or an overly conservative selected-row lower bound. Next bounded diagnostic will capture the exact matrix and compare its singular values using a separately charged validation SVD. No SVD fallback is permitted in production. A bounded scalar Jacobi alternative may resolve it, but requires a frozen numerical bracket/control test before use.
