# T238 symmetric grid-law centering prerequisite

## Question
Does the inherited cumulative-grid inverse preserve the center of an even density, and is there a narrow algebraic repair?

## Method
Written source/algebra under claim9d8db86d/shared5a879f47. Freeze original core.py60967f89 and T235's independently reviewed formal grid law. PROOF.md treats a symmetric uniform exact-real grid and distinguishes its interval-uniform interpolation from discrete endpoint sampling. Independent reviewer recover_78 in review/t238_centering_review.

## Status
Bounded source/proof phase complete. Independent written review9f6057de, published2d3ef8ee, passes all six frozen file bindings and the derivation. No execution or shared implementation change is authorized by this phase.

## Results
For positive even ordinates on N=2m uniform intervals, source-style normalized cumulative exclusion gives central CDF gap(d_m-d_N)/(2S) and interpolated mean A*d_N/S-Delta/2, withS=sum_{j=1}^N d_j. On the formal Gaussian gridA=8,N=160000, the CDF gap exceeds1/640000 without numerical evaluation. Trapezoidal interval weights restore reflection symmetry in exact arithmetic. These are conditional algebraic claims, not measured floating-array or trained-model results.

## Compute accounting
Text/Git byte reads, copies/hash and mathematical reasoning only. Zero scientific code, grid/table construction, array decoding, Gaussian integration, face/network, training, fitting, RNG or Monte Carlo calls. No source cost or performance claim.

## Claims
- [inferred, independently reviewed] The formal right-endpoint cumulative construction can violate Gaussian central symmetry.
- [inferred, independently reviewed] Trapezoidal interval weights repair that property on even grids.
- [speculative] A centered numerical construction could be useful; no target benefit or all-integrand accuracy claim is established.

## Alternatives not ruled out
Actual floating-table cancellation/rounding may alter the formal bias; the trained integrand may be insensitive to it. Alternative quadrature, density interpolation or controlled integration can address the full law. Constant density is an exact control with zero centering bias. Finite asymmetric T>2 face laws need a different analysis.

## Failure modes
Even-grid assumptions do not bind actual floating values. Source Z and inverse-law normalizers differ. Fixing mean/CDF symmetry does not match the full Gaussian density or certify finite-rule/target error. Endpoint, truncation, walker, numerical mass and cost errors remain separate. No method is declared dead.
