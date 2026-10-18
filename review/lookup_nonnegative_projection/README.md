# Fixed hidden nonnegative projection

## Question
Does enforcing the network hidden state's nonnegativity improve the existing
T201 lookup's fidelity, or only its hidden squared error?

## Method
Keep the original fit and all three fixed panels primary. Project the lookup
coordinatewise by `p=max(z,0)` without changing coefficients. `primitive.py`
computes the projected readout and before/after hidden squared errors. Initial
`controls.py` fixes ten analytic fixtures and six rejection/overflow controls.
The separately gated `saved.py` applies this unchanged projection to all192
original chunks; no lookup coefficients or model inputs are recomputed.

For actual h>=0, a negative z contributes `(z-h)^2-h^2=z^2-2zh>=0`
to the exact squared-error reduction; nonnegative z is unchanged. Summing proves
pointwise hidden SSE cannot increase. This concerns exact real values of stored
numbers; ordinary floating SSE/reduction evaluation is a separate diagnostic.
Readout labels are not ordered by hidden SSE. Controls include strict and tied
examples where network agreement worsens, plus an improving example. First-index
argmax is preserved. The cancellation fixture preserves native readout rounding;
it does not claim exact floating dot products or a universal error bound.

## Status
Bounded controls and fixed saved application complete; full saved audit pending.
Claim cce45a1b preceded implementation. h4 control/source gates405b4f81/0fd567dc
and published immutable execution gates preceded both once-only runs. T55
independently checked controls8c8f8474 and owns the separate full saved audit.
No further numerical continuation or predictor adoption is automatic.

## Results
Raw45ab5489/shared32e3aa0f retains all196608 observations. Heldout network
agreement improves by53/75 net observations, while task correctness loses56/17.
Hidden SSE decreases0.263096%/0.239776%; all paired improvements and regressions,
clipping magnitudes and contexts are retained. `SUMMARY.md` gives all three
panels. These reused data provide a descriptive follow-up, not fresh validation.

Historical control command (existing output refuses reruns):

    ulimit -v 2097152
    ulimit -t 20
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python review/lookup_nonnegative_projection/controls.py

The saved application used the same caps/runtime with `saved.py`; see
`SAVED_PROTOCOL.md` for its immutable dependencies and exact command.

## Compute accounting
No fit, neural recurrence, random input or reference computation. Projection,
SSE and readout arithmetic are diagnostic work and charged as named expression
counts, not a native count or estimator benchmark. Validation, comparisons in
argmax, control exact Fraction checks, parsing, GCD, allocation and runtime
overheads are outside that partial ledger. Main-scope CPU is.038630243 for
controls and10.535777677 for the saved application; the external process cap
also covers top-level imports/output creation/final serialization outside the
stored timing. The controls additionally compute original/lookup readouts;
the saved phase reuses those baseline arrays and computes only the new readout.
Dense dot expressions use H products plus H-1 additions (2H-1), not T196's
canonical dense2H convention. These partial category vectors cannot be silently
compared to a T196 budget or an installed microkernel count.

## Claims
[inferred] Coordinate projection weakly reduces exact hidden SSE under h>=0.
[inferred] It need not improve either network agreement or task accuracy.
[measured, full saved audit pending] The fixed projection slightly improves
network agreement but worsens task accuracy on both reused heldout panels.
[speculative] Its fixed-panel comparison may distinguish a hidden-domain
constraint error from other lookup approximation errors. No causal claim follows.

## Alternatives not ruled out
Negative predictions occur on about one quarter of rows but change fewer than
400 labels per panel; all coordinate/row clipping counts and magnitudes are
retained. Hidden SSE changes can fall in readout-insensitive directions; paired
network/task transitions expose the limited observed correction without
separating a causal input mechanism from fitted approximation error.
Other constrained fits, summaries and omitted-input mechanisms are untested;
no family is declared dead or selected for adoption.

## Failure modes
Reject nonfinite/malformed inputs and negative actual hidden states. Overflow
is preserved as failure. Source/runtime drift or an existing output directory
rejects execution; durable attempts/results retain partial failures. Floating
projection/readout/SSE behavior is finite evidence, not a full numerical proof.

The control reviewer preserved a signed-zero readout comparison failure and
narrow correction; no author calls were repeated. All146827 unclipped saved
rows retain exactly equal baseline logits under numeric equality. This finite
parity does not certify arbitrary platform/kernel behavior.
