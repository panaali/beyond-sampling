# Joint stored-matrix cofactor angle

## Question
Does the full stored-matrix angle differ materially from the saved source argument, after accounting for denominator approximation as well as numerator cancellation?

## Method
Exact rational determinants and Jacobi identity, symmetry/SPD/domain checks, inherited outward sqrt/atan enclosures. See PROTOCOL.md. Analytic controls precede any saved64 application, separately source-gated.

## Status
Claim d7ad9e62 pushed. Initial12 attempted after source gate;9 saved and3 serialization-only failures preserved b279ad68. Failed-only retry3 passed be678059 after separate source gate. Merged12 unique cases await independent saved audit; application remains gated.

## Results
None yet. Original numerator-only results remain valid in their scope.

## Compute accounting
Original12 controls .131925699 CPU plus failed-only3 retry .147994698 CPU:15 actual mathematical attempts for12 unique fixtures. Each artifact retains full operation categories, including repeated failed-attempt work. Zero native determinant/model/mass/MC calls. Exact setup and MP diagnostic costs are not scalar/native priced.

## Claims
[inferred] The atan2 denominator is sqrt(cofactor_kk*cofactor_ll−cofactor_kl²), not the asin denominator. Source om can differ from the exact stored-matrix complement minor.

## Alternatives not ruled out
Denominator error may be negligible, or source covariance construction/quadrature may dominate. This primitive cannot establish wholemass accuracy.

## Failure modes
Non-SPD/asymmetric/zero radicand and insufficient sqrt precision must remain unknown. Exact source matrix differs from its intended real covariance; libm and quadrature remain outside.

First12 control attempt b279ad68 retained:9 saved,3 serialization failures after the mathematical and diagnostic assertions passed. Python4300-digit conversion limit prevented storing the exact angle bounds. Only those3 failed records are queued for a separately pinned retry with unrestricted integer-string serialization; no primitive or fixture change and no rerun of the9 completed controls. This is an implementation failure, not evidence against the angle theorem.
