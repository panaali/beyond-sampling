# Fixed-rule Schur polynomial kernel (T160)

## Question
Can precomputed cubic cofactors replace the per-node four-dimensional inverse without materially changing the frozen quadrature values?

## Method
Claim8078e013 pushed before implementation. Keep original dual-normal inverse, covariance construction, actual rounded correlation diagonal, source determinant cutoff and24/96 rules. On Rt=(1−t)I+tC, each conditional correlation equals a cubic cofactor numerator divided by the square root of two principal3x3 minors. Evaluate minors using their nonnegative Bernstein form when C is positive semidefinite; no assumption that rounded C is exactly PSD or unit diagonal. Keep source final correlation clipping but record excursions and reject nonpositive variances explicitly. No source fallback in this initial discriminator.

First verify algebra with Fraction rational symmetric positive definite matrices and exact principal cofactors. Analytic cone controls: orthant, coordinate order and independent120-degree pairs. Deterministic near-singular covariance controls use C=(1−rho)I+rho ss^T at rho=0,.5,.95,1−2^-20,1−2^-40, s=(1,−1,1,−1). Compare individual conditional correlations against direct inverse at every fixed24/96 node. Then evaluate exactly256 evenly spaced indices floor(j*35097/255), j=0..255 from frozen seed0 exact error archive, including endpoints; no selection by observed error. Report every failure, node deviation, integrated deviation and local operation categories. Gate for proposing a separate full-corpus phase: no failures and max integrated deviation <=1e-10. This is a pragmatic numerical parity screen, not a roundoff certificate. If it fails, preserve the failure and first diagnose conditioning with independently computed cofactors before any revised implementation.

Competing explanations: inverse cost may dominate (compare explicit setup and node categories); cofactor cancellation may destroy parity near singular endpoints (fixed near-singular controls and node residuals); covariance preprocessing may dominate both (same initial C and exact algebra controls isolate this). No geometry, new model inputs, MC, positive-reference integration or seed1 replay. The frozen T149 production candidate remains unchanged.

## Status
Initial bounded controls and diagnosis complete; trained256 panel withheld after severe numerical instability. Independent review requested. Owned folder only, one thread2GB,<=20CPU review or55CPU primary phase, independent T55 review required.

## Results
`controls.py` verifies320 Fraction identities and three analytic cone masses. Near-singular controls expose a binary64 implementation failure: at rho=1−2^-20, the worst partial-correlation deviation from exact equicorrelation is1.5918956e-5; at rho=1−2^-40 it is1,732,665.7686. `diagnose.py` uses exact same-binary C and t and the closed-form conditional correlation s_k*s_l*(t*rho)/(1+2*t*rho). The source inverse errors at those nodes are1.80e-11 and5.89e-6, respectively, also not exact.

At the thinner fixture, the true numerator is about−2.16344e-23 but floating Horner evaluation gives+1.11022e-16. Exact evaluation of rounded coefficients gives−2.08072e-23: both coefficient construction and polynomial evaluation contribute, with the latter dominant. The diagonal Bernstein minor is6.40760e-23 versus exact6.49032e-23, so positive polynomial terms alone do not make rounded coefficient preprocessing accurate. All exact cofactor identities still hold. Correlation clipping would conceal the catastrophic error, so no trained panel or cost benefit is claimed.

Reproduce `controls.py` then `diagnose.py` with the project interpreter under one-thread limits. Raw controls and diagnosis are saved, including invalid raw correlations rather than clipped results.

## Compute accounting
Charge polynomial coefficient construction and evaluations, initial inverse/determinant, source comparator, controls and diagnostic node inversions separately. No inherited source node counter is a valid new kernel cost. Hardware timing and operation categories remain distinct. Recorded controls CPU.086544743 and diagnosis CPU.078900606, total.165445349. No trained subset was run; no estimator or complete portable-cost claim.

## Claims
[inferred] Schur complements provide the same real-arithmetic conditional correlation on positive definite matrices. [measured] This binary64 power-basis numerator implementation is unstable on the frozen near-singular controls and is not ready for trained replay. [speculative] Stable Bernstein numerator evaluation with accurate coefficient preprocessing or a rigorously guarded fallback may retain the algebraic cost opportunity.

## Alternatives not ruled out
Exact/compensated cofactor preprocessing, guarded source fallback, ordinary power-basis minors, Bernstein numerator representation with accurate preprocessing, other Schur evaluation order and positive integration. Failure of binary64 preprocessing is not a negative for Schur algebra.

## Failure modes
Rounded C can be indefinite, principal-minor cancellation, small negative variances, raw correlations outside unit interval, source inverse conditioning, setup overhead, exploratory panel limitations and equal totals hiding per-cone deviations.
