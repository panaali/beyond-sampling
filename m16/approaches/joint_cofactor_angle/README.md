# Joint stored-matrix cofactor angle

## Question
Does the full stored-matrix angle differ materially from the saved source argument, after accounting for denominator approximation as well as numerator cancellation?

## Method
Exact rational determinants and Jacobi identity, symmetry/SPD/domain checks, inherited outward sqrt/atan enclosures. See PROTOCOL.md. Analytic controls precede any saved64 application, separately source-gated.

## Status
Claim d7ad9e62 pushed. Initial12 controls frozen, awaiting independent source gate; not executed.

## Results
None yet. Original numerator-only results remain valid in their scope.

## Compute accounting
No numerical execution yet. Future exact determinant, interval and setup categories recorded separately; no native budget claim.

## Claims
[inferred] The atan2 denominator is sqrt(cofactor_kk*cofactor_ll−cofactor_kl²), not the asin denominator. Source om can differ from the exact stored-matrix complement minor.

## Alternatives not ruled out
Denominator error may be negligible, or source covariance construction/quadrature may dominate. This primitive cannot establish wholemass accuracy.

## Failure modes
Non-SPD/asymmetric/zero radicand and insufficient sqrt precision must remain unknown. Exact source matrix differs from its intended real covariance; libm and quadrature remain outside.

First12 control attempt b279ad68 retained:9 saved,3 serialization failures after the mathematical and diagnostic assertions passed. Python4300-digit conversion limit prevented storing the exact angle bounds. Only those3 failed records are queued for a separately pinned retry with unrestricted integer-string serialization; no primitive or fixture change and no rerun of the9 completed controls. This is an implementation failure, not evidence against the angle theorem.
