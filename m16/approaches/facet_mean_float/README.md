# T175 binary64 facet mean evaluator

## Question
Can ordinary floating arithmetic replace the MP numerical evaluation of the exact-prepared Gaussian facet means on the existing controls?

## Method
Author34 yields this numerical-only lane; their pair-moment work is separate. Reuse20 immutable exact-prepared facet certificates, including duplicate/zero/opposite constraints. Convert exact rational Gram data to binary64, then evaluate sqrt/atan2 and signed terms in the original order. Compare to saved200-digit values with separate absolute1e-12 and relative1e-8 gates; exact zero requires zero for the relative gate. No clipping or cancellation repair in this initial variant.

## Status
Claim6fb15756 pushed before implementation. Source frozen; independent gate before one20-control run. No trained-cell application yet.

## Results
Pending. All detected failures remain visible. The conversion guard detects coefficient underflow; intermediate sqrt/division/product underflow is not exhaustively flagged, so raw terms and saved-reference errors are retained. Original35 controls include12 distinct radial controls and3 invalid-preparation controls; this scope uses exactly20 facet controls and does not claim to replace the radial/preparation API.

## Compute accounting
One thread2GB20CPU review. Exact rational preparation and its comparison/GCD/bit costs remain inherited real work. New float operation categories are named-source counts, not native or complete runtime cost; integer parsing/control/library innards and diagnostics remain separate. MP/reference integration is not rerun. No speed ratio from historical timers.

## Claims
[inferred] The same real-arithmetic facet identity is evaluated. [speculative] Binary64 suffices on the fixed controls and later saved trained cells; signed cancellation may defeat relative accuracy.

## Alternatives not ruled out
Positive rearrangements, scaled conversion, compensated accumulation or a declared higher-precision fallback may rescue failures, but none is silently adopted. Lower numerical-evaluation cost alone does not establish total model or challenge accuracy.

## Failure modes
Signed facet cancellation; conversion overflow/underflow; rounded zero radicands/norms; negative outputs; unsupported exact domain; treating old preparation as free model-independent constants; confusing finite MP agreement with an outward error certificate.
