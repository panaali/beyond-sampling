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
estT uses its wide12-point rule with t=1-(1-s)^2 and the original weight and
Jacobian; retain every pair/complement ordering and normalization factor.

Evaluate these frozen arms at80 decimal digits:
1. Rounded float-derived correlation C and original stored double t/w/jac:
   isolates arithmetic evaluation of exactly the stored source formula.
2. C reconstructed from the exact binary ray entries at high precision,
   with the same stored t/w/jac: isolates correlation construction rounding.
3. Same high-precision C with high-precision12-point Gauss-Legendre nodes,
   weights and the unchanged power2 substitution: isolates stored-rule
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
3. Both remain negative and ideal12-point nodes do too: the finite rule is
   inaccurate on this fixture, given the separate positive reference.
4. An invalid correlation or singular transformed matrix blocks evaluation:
   preserve the failure and do not silently repair the model/formula.

## Status
Bounded fixture evaluation complete and independently reviewed ea53df3.
T55 reviewed formula, exact binary lifting, guards and controls before the
fixture, then acknowledged the corrected branch protocol.

## Results
The corrected source uses rmax=.9418468705031257 (<.95), determinant
1.744646238739237e-9 (>1e-12), and therefore wide12/power2.

| Evaluation | Probability | Change from source |
| --- | ---: | ---: |
| Original float source | -4.803771250228017e-7 | 0 |
| MP rounded C, rounded rule | -4.803771250093878e-7 | 1.34139331533e-17 |
| MP exact-ray C, rounded rule | -4.803771249943949e-7 | 2.84067988561e-17 |
| MP exact-ray C, ideal GL12/power2 | -4.803771249879328e-7 | 3.48689759559e-17 |
| Existing positive reference | 1.473096100363147e-11 | 4.80391855984e-7 |

All three MP arms remain negative. Increasing arithmetic precision and
reconstructing C and the same-order rule changes the result by only about
3.49e-17, versus the 4.80e-7 reference discrepancy. Thus finite quadrature
error dominates this fixture. Maximum 80/100-digit difference is3.59e-82;
that is stability evidence, not a formal MP rounding certificate. No source
asin clipping or denominator floors activate on this fixture.

The max-pair-correlation threshold does not protect this example. This is
one posthoc bug fixture, not a distributional result, a full estimator, or
a rejection of Plackett integration with a different rule or error control.

## Compute accounting
One thread,2GB, review lock, <=20CPU per phase. Count all correlation and
transformed-matrix inversions, arcsines and quadrature nodes, including
model-independent rule generation. Prior positive/capture costs remain
with T131 and are not claimed free. No sampling, geometry enumeration,
source edits, fitted tolerance or noisy-model sweep.

## Claims
[inferred] Holding nodes/covariance fixed separates rounding from finite-rule
error for that frozen numerical expression.
[measured] Finite-rule error dominates this frozen negative fixture;
arithmetic/covariance/node rounding accounts for only about3.49e-17.

## Alternatives not ruled out
Different quadrature rules/transforms, positive integration, certified error
bounds, topology defects elsewhere, and full estimator/cost tradeoffs.

## Failure modes
Changed covariance/permutation, implicit asin clipping, lost exact binary
inputs, cancellation hidden by inadequate precision, or generalizing one
posthoc diagnostic to the full captured population.

## Preserved setup failure and correction
The initial protocol assumed thin48/power3. Its explicit branch guard failed
before any high-precision fixture evaluation; `fixture_failure.json` and
commit d913169 preserve the failure. Fresh TASKS scope correction333aea4
freezes the actual source wide12/power2 branch. `rule_metadata.json` records
the source rmax, determinant guard and all original rounded nodes, weights
and Jacobians before the corrected fixture run. No numerical attribution
was made from the failed setup.

Corrected fixture CPU .156275s; metadata .011473s. The six MP arm evaluations
(three each at80 and100 digits) used72 transformed inversions/nodes,468
arcsines,72 positive-definiteness checks, two exact-ray covariance inversions
and two ideal-rule constructions. Float diagnostic adds12 inversions; the
source replay retains its1980 named operations. These are review costs,
not production-cost savings.

Reproduce the bounded phases with the repository interpreter (one thread,
2GB limit, review lock):
```sh
python m16/approaches/plackett_precision_review/pin_rule.py
python m16/approaches/plackett_precision_review/review.py controls
python m16/approaches/plackett_precision_review/review.py fixture
```
These commands use only the pinned existing source fixture and rule.

T55 independently reconstructed correlation through the ray-Gram inverse
and conditional correlations through2x2 Schur complements at100 digits.
All three arms agree within2.33e-99 and the finite-rule attribution survives
that different implementation (review ea53df3). DRI classification remains
separate; this does not certify other simplices or a full estimator.
