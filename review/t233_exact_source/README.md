# Independent T233 exact-control source review

Delegation291079cc; source15e9990e and pre-run return-durability/proof-pin repair
c0b505c3 inspected read-only. No author primitive, control, target or numerical
Gaussian calls. The inherited conditional and gauge proofs were independently
checked by78 (6c02b3aa and725d4912); this review concerns their exact-rational
implementation and bounded harness, not a second empirical test.

## Algebra/source findings

The primitive admits exact int/Fraction but rejects bool/float. point uses the
correct strict anchor gate; a=0 gives residual b+ and a=b=0 gives zero. Opposite
nonzero slopes imply the two threshold interval formula. At the anchor endpoint
x=u, b=beta(u−v), so max(beta(u−v),0) is the necessary exact point exception;
at the other endpoint x=v the residual vanishes. Coincident thresholds are zero.
The area |beta| |u−v|²/2 is a Lebesgue triangular area, never a Gaussian expected
value by itself. Positive scaling preserves both roots and weighted area when
the margin coefficient scales inversely. Negative margin envelopes reject.

The12 fixtures cover all slope/orientation and root-order combinations, narrow
and wide gaps, small slopes, coincident roots, fractional roots, both endpoint
ownership cases, and zero margin. Harness reference max(b,0)−b*1[a<0] differs
from the candidate's sign-support implementation. Its antiderivative area uses
ordered endpoints and absolute signed integral, consistent in both orientations.
There are120 point calls,60 interval calls,12 scaling calls and8 guards; nested
conditional invocations are separate from interface counts. No cost conclusion.

## Pre-run harness issues and requested repairs

Initial source allowed an arbitrary complete/pass_all review artifact without
requiring its pins to cover the actual primitive/driver/fixtures/protocol.
Requested explicit membership plus validation of every reviewer pin.

Initial save overwrote results directly; a hard interruption could truncate the
only prior record. Requested same-directory temporary write and atomic replace.

The final event checks CPU after complete/pass_all were set true. A cap failure
there would preserve contradictory completed and fatal status. Requested both
flags reset in the catch-all failure handler. These are harness issues, not a
negative scientific control outcome. No author run was executed or requested
before their repair; earlier source commits remain preserved.

## Status and exclusions

Source/resource clearance pending those narrow guards. A completed source_gate
artifact will name the final pins and exactly one allowed synthetic run. Actual
soft/hard OS limits must be finite and <=20CPU/2GiB, internal17CPU; one thread,
review lock, pinned Python, durable no-retry outcomes. Saved-output verification
belongs independently to T55. No Gaussian constant/CDF, state closure, target
performance, stable floating kernel or full computational cost is certified.
