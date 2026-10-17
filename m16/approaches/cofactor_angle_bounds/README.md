# T109 fixed-denominator cofactor angle bounds

## Question
Does the reviewed cofactor interval give a useful bound on the resulting partial angle when its denominator is held fixed at the stored binary value?

## Method
Claim5be8e715 before implementation. For x>0, atan2(y,x) has derivative x/(x²+y²); maximize it over the signed numerator interval and multiply the maximum distance from the selected point. Use exact rational arithmetic and the universal cap4 (angle difference<pi<4). For x=0 and a strictly one-sign interval the angle is constant; intervals containing the origin stay unknown. Independent proof/control gate precedes21 fixed analytic fixtures; only then reuse original64 saved cofactor intervals and stored atan2 denominators. Preserve every failed/unknown case.

## Status
Independent source gate passed; all21 frozen analytic/undefined controls pass. Exact rational proof remains separate from120-digit diagnostics. Original64 continuation completed after independent source/control gate and immutable completed-output linkage; independent saved aggregate review passes6e4862c0.

## Results
[inferred] The derivative formula gives a numerator-only error bound for each fixed nonnegative denominator. T55 independently checked the algebra and required explicit origin-undefined status. [measured] All21 frozen analytic controls and64 original saved cases pass. The maximum numerator-only bound is1.2710635397296333e-9 radians; candidate diagnostic errors improve34 cases, worsen27 and tie3. These are fixed-denominator comparisons, not whole-mass or speed results.

## Compute accounting
One thread2GB, review<=20CPU. No determinant, geometry, model, mass or MC replay. Exact rational arithmetic/conversions, sign transforms and peak numerator/denominator bits will be recorded separately; high-precision reference angle work is evaluation-only and never free production work.

## Claims
[inferred] Fixed-denominator monotonicity permits exact rational propagation without a new quadrature rule. [speculative] Original64 intervals may yield sufficiently narrow conditional angle errors to motivate later weighted propagation. No full-mass or native-speed claim.

## Alternatives not ruled out
Cancellation may make a certified determinant interval too wide; test explicit cross-zero and very small denominator controls. Source denominator/covariance/libm error may dominate; this experiment deliberately cannot resolve those errors. The original fast point can be zero for nonzero determinants, so a successful enclosure must not be described as a guarded point or guarded mass.

## Failure modes
Treating atan2(0,0) as a defined real angle, ignoring sign changes, comparing derivative error with a different denominator, inferring mass accuracy from64 nodes, uncharged exact arithmetic, or confusing ordinary high-precision diagnostics with proof of floating libm rounding.

Frozen21 controls pass with all undefined/invalid statuses retained; CPU0.004003642000000016. Reproduce with controls.py only in a fresh output-free checkout under the20CPU review lock. No determinant/model/mass work.


Original64 saved interval continuation passes after completed21/source hashes were frozen8614fc95. All64 denominators are positive and all64 rational bounds are finite; no unknowns on this saved subset. Maximum bound1.27106353973e-09 radians (label825,node85,pair1; see exact raw for authoritative index), while source/point angle errors are separate highprecision diagnostics. No determinant/model/mass calls;192 diagnostic atan2 evaluations, .024007CPU, plus192 sign transforms and256 input-dyadic conversions. Exact rational arithmetic and operand bit sizes are retained, not priced as cheap floating operations. Independent saved64 audit passed6e4862c0.


Continuation scope81b1841d is frozen in weighted_protocol.json and WEIGHTED_PROOF.md. Same eight archived cones, all768 nodes each, original64 guard records reused once. New coefficient reconstruction128 native3x3 determinants and6080 new guarded minors are explicitly charged. All stored Rt/denominator bytes must match before per-cone contraction. Seventeen exact upward-dyadic controls and independent source gate passed before execution; the continuation has completed as recorded below. This is an ideal weighted-angle numerator bound, excluding original source floating arithmetic and total mass accuracy.


## Weighted continuation (scope81b1841d)
Independent initial21/64 review6e4862c0 passes. New exact17-ceiling controls and source gate passed before the one bounded run. Checkpointe7159c42 preserves all6144 cases:6080 new guarded minors plus64 reused once; all8 reconstructed Rt and denominator arrays agree bytewise with the archived source. All6144 numerator intervals produce finite conditional angle bounds, with no unknowns on this subset. Per-cone weighted upper bounds range5.2282917647e-17 to2.0926326518e-14; exact dyadic totals in weighted_results.json are authoritative. The selected binary-weight ideal functional differs from the source sequential floating evaluation; these figures are not total mass error bounds.

Run CPU .823861874 seconds, one thread under20CPU review cap. Coefficient setup adds128 native3x3 determinants,128 norm4 calls,32 dot4 calls,8 matrix4 products and one leggauss128 setup. All rational/integer/sign categories are retained; setup internals/control/runtime remain partial. No model, geometry, source mass, quadrature-order replay or angle-reference evaluations. The17 ceiling control phase reported0.0 process CPU at timer resolution; this does not mean zero operations.

[measured] On these eight cones, the numerator-only enclosure is small after weighting, so this particular interval looseness does not obstruct a later mass guard. [inferred] This result cannot certify denominator construction, covariance, coefficient formation, libm, quadrature or the whole mass, nor establish a native speed improvement. Existing point cancellation and generic failure cases remain preserved. Independent final saved audit passes: reviewer recomputed all6144 exact Fraction determinants from stored Rt, every enclosure/sign/derivative bound, selected-weight ceiling and eight totals;1.095CPU, no native/mass calls. Review commit pending. Next useful separate lane is exact T5 Gauss-Legendre node/weight enclosures; no further T4 mass replay is proposed here.
