# T109 fixed-denominator cofactor angle bounds

## Question
Does the reviewed cofactor interval give a useful bound on the resulting partial angle when its denominator is held fixed at the stored binary value?

## Method
Claim5be8e715 before implementation. For x>0, atan2(y,x) has derivative x/(x²+y²); maximize it over the signed numerator interval and multiply the maximum distance from the selected point. Use exact rational arithmetic and the universal cap4 (angle difference<pi<4). For x=0 and a strictly one-sign interval the angle is constant; intervals containing the origin stay unknown. Independent proof/control gate precedes21 fixed analytic fixtures; only then reuse original64 saved cofactor intervals and stored atan2 denominators. Preserve every failed/unknown case.

## Status
Independent source gate passed; all21 frozen analytic/undefined controls pass. Exact rational proof remains separate from120-digit diagnostics. Original64 continuation completed after independent source/control gate and immutable completed-output linkage; saved aggregate review pending.

## Results
[inferred] The derivative formula gives a numerator-only error bound for each fixed nonnegative denominator. T55 independently checked the algebra and required explicit origin-undefined status. No measured results yet.

## Compute accounting
One thread2GB, review<=20CPU. No determinant, geometry, model, mass or MC replay. Exact rational arithmetic/conversions, sign transforms and peak numerator/denominator bits will be recorded separately; high-precision reference angle work is evaluation-only and never free production work.

## Claims
[inferred] Fixed-denominator monotonicity permits exact rational propagation without a new quadrature rule. [speculative] Original64 intervals may yield sufficiently narrow conditional angle errors to motivate later weighted propagation. No full-mass or native-speed claim.

## Alternatives not ruled out
Cancellation may make a certified determinant interval too wide; test explicit cross-zero and very small denominator controls. Source denominator/covariance/libm error may dominate; this experiment deliberately cannot resolve those errors. The original fast point can be zero for nonzero determinants, so a successful enclosure must not be described as a guarded point or guarded mass.

## Failure modes
Treating atan2(0,0) as a defined real angle, ignoring sign changes, comparing derivative error with a different denominator, inferring mass accuracy from64 nodes, uncharged exact arithmetic, or confusing ordinary high-precision diagnostics with proof of floating libm rounding.

Frozen21 controls pass with all undefined/invalid statuses retained; CPU0.004003642000000016. Reproduce with controls.py only in a fresh output-free checkout under the20CPU review lock. No determinant/model/mass work.


Original64 saved interval continuation passes after completed21/source hashes were frozen8614fc95. All64 denominators are positive and all64 rational bounds are finite; no unknowns on this saved subset. Maximum bound1.27106353973e-09 radians (label825,node85,pair1; see exact raw for authoritative index), while source/point angle errors are separate highprecision diagnostics. No determinant/model/mass calls;192 diagnostic atan2 evaluations, .024007CPU, plus192 sign transforms and256 input-dyadic conversions. Exact rational arithmetic and operand bit sizes are retained, not priced as cheap floating operations. Independent saved64 audit pending.
