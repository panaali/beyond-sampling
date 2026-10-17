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
