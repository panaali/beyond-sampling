# Fixed-rule Schur polynomial kernel (T160)

## Question
Can precomputed cubic cofactors replace the per-node four-dimensional inverse without materially changing the frozen quadrature values?

## Method
Claim8078e013 pushed before implementation. Keep original dual-normal inverse, covariance construction, actual rounded correlation diagonal, source determinant cutoff and24/96 rules. On Rt=(1−t)I+tC, each conditional correlation equals a cubic cofactor numerator divided by the square root of two principal3x3 minors. Evaluate minors using their nonnegative Bernstein form when C is positive semidefinite; no assumption that rounded C is exactly PSD or unit diagonal. Keep source final correlation clipping but record excursions and reject nonpositive variances explicitly. No source fallback in this initial discriminator.

First verify algebra with Fraction rational symmetric positive definite matrices and exact principal cofactors. Analytic cone controls: orthant, coordinate order and independent120-degree pairs. Deterministic near-singular covariance controls use C=(1−rho)I+rho ss^T at rho=0,.5,.95,1−2^-20,1−2^-40, s=(1,−1,1,−1). Compare individual conditional correlations against direct inverse at every fixed24/96 node. Then evaluate exactly256 evenly spaced indices floor(j*35097/255), j=0..255 from frozen seed0 exact error archive, including endpoints; no selection by observed error. Report every failure, node deviation, integrated deviation and local operation categories. Gate for proposing a separate full-corpus phase: no failures and max integrated deviation <=1e-10. This is a pragmatic numerical parity screen, not a roundoff certificate. If it fails, preserve the failure and first diagnose conditioning with independently computed cofactors before any revised implementation.

Competing explanations: inverse cost may dominate (compare explicit setup and node categories); cofactor cancellation may destroy parity near singular endpoints (fixed near-singular controls and node residuals); covariance preprocessing may dominate both (same initial C and exact algebra controls isolate this). No geometry, new model inputs, MC, positive-reference integration or seed1 replay. The frozen T149 production candidate remains unchanged.

## Status
Protocol frozen; controls pending. Owned folder only, one thread2GB,<=20CPU review or55CPU primary phase, independent T55 review required.

## Results
Pending. Scripts and raw failures will be retained. Reproduce with the project interpreter under one-thread limits.

## Compute accounting
Charge polynomial coefficient construction and evaluations, initial inverse/determinant, source comparator, controls and diagnostic node inversions separately. No inherited source node counter is a valid new kernel cost. Hardware timing and operation categories remain distinct. Initial local subset only; no estimator or complete portable-cost claim.

## Claims
[inferred] Schur complements provide the same real-arithmetic conditional correlation on positive definite matrices. [speculative] Precomputation may reduce production cost without changing quadrature bias.

## Alternatives not ruled out
Exact/compensated cofactor preprocessing, guarded source fallback, ordinary power-basis minors, other Schur evaluation order and positive integration. Failure of binary64 preprocessing is not a negative for Schur algebra.

## Failure modes
Rounded C can be indefinite, principal-minor cancellation, small negative variances, raw correlations outside unit interval, source inverse conditioning, setup overhead, exploratory panel limitations and equal totals hiding per-cone deviations.
