# Guarded 3x3 cofactors: bounded cheap-mass primitive

## Question

Can a fixed small determinant primitive provide an explicit error enclosure at lower declared arithmetic cost than the currently used general determinant path? This is a bounded subgoal of T109, claimed and pushed ed647955 before implementation, with T11580 linked to prevent duplication. The original300-unit cone-mass goal remains open.

## Method

Freeze binary64 round-to-nearest, fixed nonfused Laplace expression with14 arithmetic operations. For finite stored entries either zero or with absolute value in[2^-100,2^100], use the same positive14-operation tree on absolute values to approximate the permanent. Let u=2^-53, Phat be that positive result, and E=2^-50 Phat. Outward endpoints around Dhat-E and Dhat+E enclose the exact determinant of the stored matrix. See PROOF.md for operation order, range and error assumptions. Out-of-range finite inputs use an explicitly charged exact-rational fallback; nonfinite inputs are unknown. No silent replacement of shared code.

Freeze the25 analytic matrices in PROTOCOL.md before implementation. Independent exact determinant reference uses the six-term permutation formula, distinct from the implementation's Laplace tree. Preserve point inaccuracies and cases with intervals containing zero, rather than calling all finite enclosures an accuracy success. Source/theorem review precedes execution; independent saved-results review precedes any existing cofactor sample.

## Status

Frozen25 analytic controls complete after independent theorem/source gate. Independent saved-results audit cb751402 passes all25 exact Gaussian-elimination references, endpoints and category counts. No trained cofactors, mass or model calls.

## Results

All22 finite controls enclose the exact six-permutation rational reference:17 use the fast enclosure and5 the exact fallback. The three nonfinite inputs correctly remain unknown. Nine fast enclosures contain zero, so containment alone does not resolve determinant sign.

The raw point formula returns zero for four nonzero exact determinants: near-rank-one deltas2^-50 and2^-40, and the2^99/2^-99 scaled delta2^-50 matrices. These are preserved cancellation failures of the point formula. The delta2^-26 point retains its nonzero determinant, but its conservative interval still includes zero. The finite underflow fallback also has a zero binary64 point; its directed interval contains the exact nonzero result.

The dense nonzero fast path uses31 arithmetic operations,9 absolute values and28 comparisons, plus9 finite checks,2 nextafter calls and9 float conversions recorded separately. Exact fallbacks use14 rational arithmetic operations and9 exact conversions, with maximum observed numerator3070bits and denominator2146bits. Rational gcd/normalization, interpreter work and native interiors are unpriced. Fallbacks are never called free.

Prior native review T11579 gives104 named numerical units for one installed3x3 general determinant path, excluding opaque categories. This new source-expression inventory uses a different convention and establishes neither native cost nor speed. No cost ratio is inferred.

Run the frozen controls with `/workspace/clean-neurons/.venv/bin/python controls.py` in this folder under the one-thread2GB review lock. The script refuses to overwrite results.json. Saved source hashes, input/output hexadecimal values, exact fractions and every category count are retained. Recorded control/reference CPU .000142111 seconds excludes imports and final serialization; it is audit accounting, not an estimator timing comparison.

## Compute accounting

One thread,2GB, at most20CPU seconds per primitive cycle under the review lock. Count validation, absolute values, outward rounding and exact fallback separately from the14 point operations. Exact rational operation counts and operand bit lengths are recorded, not assigned zero cost. No model, mass, MC, quadrature or training runs in this phase.

## Claims

[inferred, pending implementation] The stated guard can enclose the stored determinant under its explicit floating-point assumptions.

[speculative] A compiled or vectorized guarded primitive may reduce total source cost. The bounded Python prototype cannot establish that claim or downstream mass accuracy.

## Alternatives not ruled out

1. Point arithmetic may lose essentially every significant digit on near-singular cofactors. Cheapest discriminator: exact rational near-rank-one fixtures, retaining both the point error and enclosure width.
2. Validation, outward rounding or exact fallbacks may erase the arithmetic saving. Cheapest discriminator: per-category operation records and fallback bit lengths on the frozen controls, followed only after review by a frozen existing cofactor sample.
3. A good determinant enclosure may still produce a poor final atan2/mass result. No downstream substitution is allowed in this phase; mass propagation is a separate task.

## Failure modes

Fused/reassociated operations, non-default rounding, unsupported input ranges, underflow/overflow, nonfinite data, treating an interval containing zero as an accurate relative approximation, uncharged fallback/control work, or transferring scalar-source counts to the installed native implementation. sys.float_info.rounds records the process environment declaration; it does not protect against a later external floating-environment change. Prior T160 cancellation and T144 interval-cost evidence remains intact.
