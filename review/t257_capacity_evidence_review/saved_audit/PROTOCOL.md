# Independent fixed T257 saved arithmetic audit

Delegation6d054702. Producer raw9f04b70c; source444750e9 and gate516ee0da.
One review-lock run after independent source clearance and published immutable
gate: actual20/20CPU,2GiB/2GiB, internal16CPU. Stdlib only, no author imports,
model, weights, RNG or new accuracy evaluation. Preserve partials/failures;
no retry or overwrite.

Check all six evaluation records,64x5 curve pairs,27 printed fields, literal
spread flags and six producer events. The reviewer reconstructs error sums
using a common integer denominator of stored binary floats. Blind fractions
and table differences are exact. Nominal SE is bracketed by integer square
root on a400-bit dyadic grid, independently of producer Decimal100 arithmetic.
The rounded nominal Decimal and reported difference must agree with their
mathematical values to relative1e-98; this conservative bound covers two
100-significant-digit rounding operations and is tiny beside the frozen8ULP
test. The dyadic bracket must determine one binary64 rounding and the worst
endpoint discrepancy from the stored SE must satisfy the original8ULP bound.
No uncertainty or MC-law guarantee follows from nominal-SE consistency.

Exact stored residuals and original allowances2^-50/2^-48/halfdecimal+ULP are
retained. Rounded zoo-spread flags are diagnostics, never acceptance claims.
No implied integer hit recovery, pair covariance, checkpoint mapping, CUDA
parity, runtime verification, distributional equivalence or capacity ceiling.

Expected reviewer progress is11 events: start,pins,training,six individual
evaluation records,table,complete. Producer successful six-event sequence is
checked separately. All output/source/protocol pins are immutable; a new
producer handoff may be published without changing these original inputs.
