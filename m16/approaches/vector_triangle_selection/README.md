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
The monotonicity argument is in real arithmetic; it is not a universal
floating-point monotonicity certificate. Exact metadata agreement is tested
on the frozen finite controls, with explicit scalar checks/fallback retained.

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
Initial bounded pilot complete after independent T55 source review9f3681a
and original T116 owner review. A separately frozen arithmetic-attribution
rescue is proposed; the initial candidate is retained unchanged.

## Results
Pending. This task does not change the integral or claim higher accuracy.

## Exact-radial compatibility gate and rescue controls
The first batched radial implementation differs on7 of9309 original cases;
qmin/midpoint/ratio differences reach1.12e-16. This fails the frozen exact
metadata gate despite the small numerical differences. radial_probe.json
retains all7 cases. The predeclared scalar-radial/vector-trial rescue is used;
there is no trained pilot of the failed radial variant.

The guarded rescue preserves all9309 complete metadata dictionaries and
output values exactly, across all10 orders. Thirty exact/adjacent tolerance
fixtures preserve both selected orders and failure behavior, including scalar
rescue. Empty batches and the single-rule node cap pass. It executes93090
vector trials and18187 scalar selected/previous checks, with no full scalar
rescue on the ordinary panel. Original charges plus all additional work
are retained. The control run is slower (.3023 versus .2230CPU), a preliminary
timing observation only. Complete cold64-cone pilot awaits independent source
review; no speedup is inferred from vectorization.

## Initial complete pilot
All64 cone values and statistics remain exactly identical to the original
grouped implementation. Twelve complete calls, including two correctness
passes and five fresh alternating timings per variant, cost5.066CPU.
Median complete time grows from .381970 to .451457CPU, ratio1.181918:
this guarded scalar-radial implementation is18.2% slower on the frozen pilot.
It is not a failure of vectorization in general. Pilot values, counters and
timings remain in pilot.json; no full integration was run.

## Frozen arithmetic-attribution rescue
Scope64dc12f pushed first. On exactly the same9309 triangles, record scalar
and batched sum, norm, normalized center, center projection, squared center
projection, sign projection/norm/bound, upper bound and final radial tuple.
Competing explanations are changed reduction order and Python float `**2`
versus the array square operation; seven mismatches alone distinguish neither.
If all other relevant intermediates agree and replacing only squared center
projection by the original scalar expression restores all radial metadata,
use that minimal scalar operation in a distinct rescue implementation.
Otherwise retain the failed attribution and do not proceed with that rescue.

Preserve the initial selector and pilot files. A successful isolated rescue
must repeat all original/threshold/empty/nodecap controls and then the same
cold pilot with fresh timing, after independent source review. It does not
relax near-threshold safeguards or certify floating behavior universally.

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
