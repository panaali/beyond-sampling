# T124: guarded vectorized triangle-rule selection

## Question
Can positive triangle-rule selection be batched while preserving the exact
scalar decisions and metadata of the T116 implementation?

## Method
Claim df82455 pushed first. Pin the T116 grouped source and dependencies;
keep positive-node grouping unchanged. Its 9309 frozen controls exercise
all10 orders and thin/degenerate triangles. Use the same64-cone T106 pilot
only after independent source/control review. Never modify source folders.

First test whether batched radial setup reproduces the scalar qmin/qmax,
midpoint and ratio exactly on those existing controls. Scalar uses a mix
of sum, dot, matvec, min and max operations; batched BLAS may change their
rounding. Any difference is retained as an implementation limitation, not
ignored because it is small. A failed exact radial check triggers the
predeclared rescue: keep radial setup scalar, vectorize only order trials.

For vectorized trials use the same10 orders, bound formula and n<=max_order.
Guard nonfinite/invalid domains and every near threshold (next_ratio=1 or
error=tolerance) by scalar selection. Independently recompute the selected
bound and the previous permitted-order bound using the original scalar
expression. Accept only if selected passes and previous fails (or does not
converge); otherwise call the original scalar selector. In its convergent
regime, the positive binomial numerator decreases and denominator increases
with n, so this adjacent check certifies minimality, not just feasibility.
Return the original scalar radial metadata and selected scalar error.

All original charges remain; add vector arithmetic, guards, exact checks and
fallback costs. Timing may improve or worsen; no FLOP-saving inference.
Reproduce complete metadata, node arrays and values on original controls,
then add explicit threshold fixtures at the scalar bound and adjacent
floating tolerances, node-cap rejection and empty batches. Preserve failure
artifacts and do not broaden a scoped implementation result to the family.

## Alternatives and falsifiers
1. Trial dispatch dominates: vector selection improves complete pilot CPU.
2. Radial setup dominates: scalar preservation erases any trial savings.
3. Changed floating evaluation alters decisions/metadata: exact controls and
   selected/previous scalar checks reject the candidate.
4. Integration dominates: selection improvement need not help full runtime;
   report complete and component timings independently.

## Status
Protocol frozen; source/control experiment pending.

## Results
Pending. This task does not change the integral or claim higher accuracy.

## Compute accounting
One thread,2GB, review lock, <=20CPU per bounded phase. Preserve original
conservative allowance and explicitly add every batch/guard/scalar-rescue
cost. Five fresh complete calls per variant, alternating order, for pilot
medians only if all correctness gates pass. Same64 cones, no new full run.

## Claims
[inferred] Selected/previous scalar verification can preserve minimal order.
[speculative] Batched trial evaluation improves complete local CPU.

## Alternatives not ruled out
Native radial setup, batched CDF primitives, better positive bounds and
different triangle integration methods.

## Failure modes
BLAS reduction rounding, near-threshold order changes, underflow/nonfinite
error values, incomplete metadata accounting, and timing overhead exceeding
the work saved.
