# T175 binary64 facet mean evaluator

## Question
Can ordinary floating arithmetic replace the MP numerical evaluation of the exact-prepared Gaussian facet means on the existing controls?

## Method
Author34 yields this numerical-only lane; their pair-moment work is separate. Reuse20 immutable exact-prepared facet certificates, including duplicate/zero/opposite constraints. Convert exact rational Gram data to binary64, then evaluate sqrt/atan2 and signed terms in the original order. Compare to saved200-digit values with separate absolute1e-12 and relative1e-8 gates; exact zero requires zero for the relative gate. No clipping or cancellation repair in this initial variant.

## Status
Claim6fb15756 pushed before implementation. Source92cfb601 independently gated by910, fixed20 baseline completeddd6d3129; baseline saved audit4895c343 confirms all thresholds and exact zero summation error. No trained-cell application.

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

## Frozen control rescue
Claim347397c9 and source3ea48dd1 independently gated by910 before execution. Candidate-only heuristic triggers actualMP80 evaluation for failed/nonpositive prepared float outputs or64u times ordered absolute-term sum exceeding1e-10 times absolute output (u=2^-53). Exact zero/opposite statuses bypass. This is not a conversion/libm/arithmetic error certificate.

Resultafac123f:20/20 finite, absolute and relative gates pass, with2 MP80 evaluations and20 baseline float rows reused exactly once; total .004008970CPU. No new exact preparation, angular reference, model or MC calls. Original narrow40 baseline failure remains immutable. Exact preparation, conversion, selection, MP imports/evaluation and diagnostics are real work; no MP-timer speed ratio.

Command: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python m16/approaches/facet_mean_float/rescue_controls.py` with20CPU/2GB limits in a fresh output directory. Saved rescue auditdd19f41b passes all20 triggers/errors and24 rational decodes. Next544 scope was separately publishedb38a6e2d and source47e4faff independently gated by910.

## Frozen saved-model application
Resultdbdfcb7d evaluates all544 exact-prepared H16T4seed3 cells once, no preparation or angular-reference replay. The unchanged heuristic uses155 MP80 evaluations and1860 rational decodes; all544 cell absolute1e-12/relative1e-8 and16 pooled mean absolute1e-10/relative1e-8 gates pass. Maximum cell absolute/relative error2.538e-15/2.055e-12 and pooled absolute1.873e-15. Whole phase .151824240CPU including saved-record diagnostics/output; individual candidate timer sum .074684817 is resolution-limited and not a complete performance measurement.

Direct floating baseline retains31 cell-relative failures (513/544 pass), while every cell absolute gate passes. Retrospective saved-only pooled diagnosticb6ff5577 shows direct floating also passes all16 pooled thresholds, maximum absolute error2.169e-14. Thus fallback repairs local relative accuracy but is **not shown necessary for this model’s pooled tolerance**. No output-accuracy/covariance/downstream or general-error conclusion follows.

Inherited exact preparation includes50592mul/22848add/8160sub, network19008mul/17412add plus source-declared comparisons/slope work; rational/GCD/native work is not fully priced. Numerical partial counts are in summary.json and exclude selection/control/MP/native internals. Model-dependent exact preparation is never free precomputation.

Command: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python m16/approaches/facet_mean_float/trained.py` with20CPU/2GB limits, fresh output directory. Saved-only summary command uses the same prefix with `m16/approaches/facet_mean_float/summarize.py`. No completed runs should be repeated. Independent544 audit8f922a46 passes all decisions/errors,155MP80 term records,1860decodes and1105progress records without replay.
