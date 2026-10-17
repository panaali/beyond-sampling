# T175 binary64 facet mean evaluator

## Question
Can ordinary floating arithmetic replace the MP numerical evaluation of the exact-prepared Gaussian facet means on the existing controls?

## Method
Author34 yields this numerical-only lane; their pair-moment work is separate. Reuse20 immutable exact-prepared facet certificates, including duplicate/zero/opposite constraints. Convert exact rational Gram data to binary64, then evaluate sqrt/atan2 and signed terms in the original order. Compare to saved200-digit values with separate absolute1e-12 and relative1e-8 gates; exact zero requires zero for the relative gate. No clipping or cancellation repair in this initial variant.

## Status
Claim6fb15756 pushed before implementation. Source92cfb601 independently gated by910, fixed20 baseline completeddd6d3129; saved audit pending. No trained-cell application.

## Results
All20 finite and absolute1e-12 gates pass;19/20 relative1e-8 gates pass. narrow40 has absolute error1.1170493265907814e-17 and relative2.4347568071667757e-4, from signed terms near±.1275 yielding4.5868e-14. This is a real failed relative gate, not a whole-family negative. Timer returned0 (below resolution, not free). All detected failures remain visible. The conversion guard detects coefficient underflow; intermediate sqrt/division/product underflow is not exhaustively flagged, so raw terms and saved-reference errors are retained. Original35 controls include12 distinct radial controls and3 invalid-preparation controls; this scope uses exactly20 facet controls and does not claim to replace the radial/preparation API.

## Compute accounting
One thread2GB20CPU review. Exact rational preparation and its comparison/GCD/bit costs remain inherited real work. New float operation categories are named-source counts, not native or complete runtime cost; integer parsing/control/library innards and diagnostics remain separate. MP/reference integration is not rerun. No speed ratio from historical timers.

## Claims
[inferred] The same real-arithmetic facet identity is evaluated. [speculative] Binary64 suffices on the fixed controls and later saved trained cells; signed cancellation may defeat relative accuracy.

## Alternatives not ruled out
Positive rearrangements, scaled conversion, compensated accumulation or a declared higher-precision fallback may rescue failures, but none is silently adopted. Lower numerical-evaluation cost alone does not establish total model or challenge accuracy.

## Failure modes
Signed facet cancellation; conversion overflow/underflow; rounded zero radicands/norms; negative outputs; unsupported exact domain; treating old preparation as free model-independent constants; confusing finite MP agreement with an outward error certificate.

Run command: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python m16/approaches/facet_mean_float/controls.py` with `ulimit -v 2500000; ulimit -t 20` in a fresh output directory. Existing results refuse overwrite. A separately frozen cancellation-triggered MP fallback remains an alternative; no such numerical call has occurred.
