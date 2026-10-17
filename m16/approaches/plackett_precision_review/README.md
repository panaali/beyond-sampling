# T137: high-precision cause review of one negative Gaussian mass

## Question
T131 finds a concrete trained simplex whose unchanged Plackett implementation
returns a negative probability while an independent positive rule returns a
small positive value. Is the discrepancy primarily finite quadrature error,
rounded covariance construction, or floating evaluation of the same rule?

## Method
Claim e0c135e pushed first, coordinated with T131 owner. Use only its existing
posthoc diagnostic fixture672. Pin the fixture, source snapshot and captured
ray corpus. Do not select a new example or change its ray permutation. Source
estT uses its thin48-point rule with t=1-(1-s)^3 and the original weight and
Jacobian; retain every pair/complement ordering and normalization factor.

Evaluate these frozen arms at80 decimal digits:
1. Rounded float-derived correlation C and original stored double t/w/jac:
   isolates arithmetic evaluation of exactly the stored source formula.
2. C reconstructed from the exact binary ray entries at high precision,
   with the same stored t/w/jac: isolates correlation construction rounding.
3. Same high-precision C with high-precision48-point Gauss-Legendre nodes,
   weights and the unchanged power3 substitution: isolates stored-rule
   rounding without changing order, transformation or formula.

Report constant arcsine term, integral term, final mass and differences for
each arm. Record every argument outside [-1,1] or nonpositive covariance
pivot as an explicit failure; do not silently introduce clipping in the
high-precision calculation. The unchanged float source's clipping/guards
remain visible and are not reinterpreted as exact probability arithmetic.

First verify formula/normalization on identity correlation (1/16) and a
block-independent correlation with two nonzero2x2 blocks, whose orthant
probability is the product of two bivariate arcsine formulas. Review source
before fixture evaluation. Repeat the same fixed-rule fixture at100 digits
only to check precision stability; no higher quadrature orders or tuned
transformations. Compare the preexisting positive GL32 result and its
independently reviewed conditional integration bound, excluding floating
and geometric uncertainty. No new positive-rule integration is needed.

## Alternatives and falsifiers
1. Same-rule high-precision arithmetic removes the discrepancy: rounding
   dominates this fixture; do not attribute it to quadrature.
2. High-precision covariance removes the discrepancy: preprocessing dominates.
3. Both remain negative and ideal48-point nodes do too: the finite rule is
   inaccurate on this fixture, given the separate positive reference.
4. An invalid correlation or singular transformed matrix blocks evaluation:
   preserve the failure and do not silently repair the model/formula.

## Status
Protocol frozen before implementation. Independent source/control review
required before fixture evaluation.

## Results
Pending. This is one bug fixture, not a full mass or accuracy estimator.

## Compute accounting
One thread,2GB, review lock, <=20CPU per phase. Count all correlation and
transformed-matrix inversions, arcsines and quadrature nodes, including
model-independent rule generation. Prior positive/capture costs remain
with T131 and are not claimed free. No sampling, geometry enumeration,
source edits, fitted tolerance or noisy-model sweep.

## Claims
[inferred] Holding nodes/covariance fixed separates rounding from finite-rule
error for that frozen numerical expression.
[speculative] Finite-rule error is the dominant cause of this negative mass.

## Alternatives not ruled out
Different quadrature rules/transforms, positive integration, certified error
bounds, topology defects elsewhere, and full estimator/cost tradeoffs.

## Failure modes
Changed covariance/permutation, implicit asin clipping, lost exact binary
inputs, cancellation hidden by inadequate precision, or generalizing one
posthoc diagnostic to the full captured population.
