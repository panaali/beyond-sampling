# Guarded 3x3 cofactors: bounded cheap-mass primitive

## Question

Can a fixed small determinant primitive provide an explicit error enclosure at lower declared arithmetic cost than the currently used general determinant path? This is a bounded subgoal of T109, claimed and pushed ed647955 before implementation, with T11580 linked to prevent duplication. The original300-unit cone-mass goal remains open.

## Method

Freeze binary64 round-to-nearest, fixed nonfused Laplace expression with14 arithmetic operations. For finite stored entries either zero or with absolute value in[2^-100,2^100], use the same positive14-operation tree on absolute values to approximate the permanent. Let u=2^-53, Phat be that positive result, and E=2^-50 Phat. Outward endpoints around Dhat-E and Dhat+E enclose the exact determinant of the stored matrix. See PROOF.md for operation order, range and error assumptions. Out-of-range finite inputs use an explicitly charged exact-rational fallback; nonfinite inputs are unknown. No silent replacement of shared code.

Freeze the25 analytic matrices in PROTOCOL.md before implementation. Independent exact determinant reference uses the six-term permutation formula, distinct from the implementation's Laplace tree. Preserve point inaccuracies and cases with intervals containing zero, rather than calling all finite enclosures an accuracy success. Source/theorem review precedes execution; independent saved-results review precedes any existing cofactor sample.

## Status

Protocol and theorem frozen. No controls or trained cofactors evaluated yet. T55 reviewed the proposed range/error derivation; implementation gate remains pending.

## Results

None yet. Prior native review T11579 gives104 named numerical units for one installed3x3 general determinant path, excluding opaque categories. A source-expression count for a new Python primitive is a different convention and does not establish native cost or speed.

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

Fused/reassociated operations, non-default rounding, unsupported input ranges, underflow/overflow, nonfinite data, treating an interval containing zero as an accurate relative approximation, uncharged fallback/control work, or transferring scalar-source counts to the installed native implementation. Prior T160 cancellation and T144 interval-cost evidence remains intact.
