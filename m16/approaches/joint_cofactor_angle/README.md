# Joint stored-matrix cofactor angle

## Question
Does the full stored-matrix angle differ materially from the saved source argument, after accounting for denominator approximation as well as numerator cancellation?

## Method
Exact rational determinants and Jacobi identity, symmetry/SPD/domain checks, inherited outward sqrt/atan enclosures. See PROTOCOL.md. Analytic controls precede any saved64 application, separately source-gated.

## Status
Claim d7ad9e62 pushed. Initial12 attempted after source gate;9 saved and3 serialization-only failures preserved b279ad68. Failed-only retry3 passed be678059 after separate source gate. Merged12 unique controls independently pass95bacea8. The separately gated64 application passed c8c34973, all64 bounded; independent saved application review36ace1ce passes.

## Results
All64 exact stored matrices pass symmetry/SPD/domain tests. The largest absolute exact mathematical source-argument angle discrepancy is approximately3.79498704420e-11 radians; denominator-only max2.05608602761e-11; candidate max3.81232359412e-11. Exact rational intervals are authoritative in application.json. These are argument-angle errors, not the source libm output or fullmass errors. Candidate improvement is not established. Saved-only extrema summary2f33db74 and independent application/summary audit36ace1ce pass.

## Compute accounting
Original12 controls .131925699 CPU plus failed-only3 retry .147994698 CPU:15 actual mathematical attempts for12 unique fixtures. Each artifact retains full operation categories, including repeated failed-attempt work. Zero native determinant/model/mass/MC calls. Exact setup and MP diagnostic costs are not scalar/native priced.

## Claims
[inferred] The atan2 denominator is sqrt(cofactor_kk*cofactor_ll−cofactor_kl²), not the asin denominator. Source om can differ from the exact stored-matrix complement minor.

## Alternatives not ruled out
Denominator error may be negligible, or source covariance construction/quadrature may dominate. This primitive cannot establish wholemass accuracy.

## Failure modes
Non-SPD/asymmetric/zero radicand and insufficient sqrt precision must remain unknown. Exact source matrix differs from its intended real covariance; libm and quadrature remain outside.

First12 control attempt b279ad68 retained:9 saved,3 serialization failures after the mathematical and diagnostic assertions passed. Python4300-digit conversion limit prevented storing the exact angle bounds. Only those3 failed records are queued for a separately pinned retry with unrestricted integer-string serialization; no primitive or fixture change and no rerun of the9 completed controls. This is an implementation failure, not evidence against the angle theorem.

Application11.776156281CPU,64 exact stored-matrix cases, zero native determinant/source-mass/model/MC calls. Original15 control attempts remain in total accounting. Exact permutation determinants, interval operations and peak integer bits are recorded; indexing/comparison/conversion/serialization and interpreter costs are not assigned native operation prices.

Bounded phase complete after independent36ace1ce. Root/Claude INDEX registration requested. No full6144 expansion is proposed from this finite result. The worker advances the existing Ready T184 original H4/5/6 source-cost coverage after fresh ownership check and a pushed claim; higher T5/Hopf lanes remain with their current owners.
