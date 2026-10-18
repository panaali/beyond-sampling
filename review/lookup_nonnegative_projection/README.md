# Fixed hidden nonnegative projection

## Question
Does enforcing the network hidden state's nonnegativity improve the existing
T201 lookup's fidelity, or only its hidden squared error?

## Method
Keep the original fit and all three fixed panels primary. Project the lookup
coordinatewise by `p=max(z,0)` without changing coefficients. `primitive.py`
computes the projected readout and before/after hidden squared errors. Initial
`controls.py` fixes ten analytic fixtures and six rejection/overflow controls.
No trained application is authorized by this preparation phase.

For actual h>=0, a negative z contributes `(z-h)^2-h^2=z^2-2zh>=0`
to the exact squared-error reduction; nonnegative z is unchanged. Summing proves
pointwise hidden SSE cannot increase. This concerns exact real values of stored
numbers; ordinary floating SSE/reduction evaluation is a separate diagnostic.
Readout labels are not ordered by hidden SSE. Controls include strict and tied
examples where network agreement worsens, plus an improving example. First-index
argmax is preserved. The cancellation fixture preserves native readout rounding;
it does not claim exact floating dot products or a universal error bound.

## Status
Claim cce45a1b published before implementation. Source/control preparation only;
h4 independently reviews source, a separately published execution gate precedes
the sole16-control run. Independent saved audit and a new source/resource gate
are required before any original-panel contraction.

## Results
No calls run. Planned command only after gate publication:

    ulimit -v 2097152
    ulimit -t 20
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python review/lookup_nonnegative_projection/controls.py

## Compute accounting
No fit, neural recurrence, random input or reference computation. Projection,
SSE and readout arithmetic are diagnostic work and charged as named expression
counts, not a native count or estimator benchmark. Validation, comparisons in
argmax, control exact Fraction checks, parsing, GCD, allocation and runtime
overheads are outside that partial ledger and inside research CPU timing.
The controls additionally compute original/lookup readouts; later saved-panel
scope must explicitly count any reconstruction rather than call it free.
Dense dot expressions use H products plus H-1 additions (2H-1), not T196's
canonical dense2H convention. These partial category vectors cannot be silently
compared to a T196 budget or an installed microkernel count.

## Claims
[inferred] Coordinate projection weakly reduces exact hidden SSE under h>=0.
[inferred] It need not improve either network agreement or task accuracy.
[speculative] Its fixed-panel comparison may distinguish a hidden-domain
constraint error from other lookup approximation errors. No causal claim follows.

## Alternatives not ruled out
The original fit may already avoid consequential negative predictions; retain
all coordinate/row clipping counts and magnitudes in a later fixed comparison.
Hidden SSE improvements may occur mainly in readout-insensitive directions;
compare all paired network/task label changes, retaining worsening outcomes.
Other constrained fits, summaries and omitted-input mechanisms are untested;
no family is declared dead or selected for adoption.

## Failure modes
Reject nonfinite/malformed inputs and negative actual hidden states. Overflow
is preserved as failure. Source/runtime drift or an existing output directory
rejects execution; durable attempts/results retain partial failures. Floating
projection/readout/SSE behavior is finite evidence, not a full numerical proof.

## Fixed controls completed

The sole published-gate run passed all10 analytic fixtures and6 guards. It retained all11 input rows, including strict and tie-based agreement worsening, and exact coordinate SSE records. Research CPU 0.03863024300000001 seconds; no trained/model/fit/RNG calls. Raw `controls/results.json` and durable progress are immutable. Independent saved review remains pending; this does not authorize a trained application.

## Independent controls and saved source

Saved controls independently pass8c8f8474/sharedf4a8964d; reviewer-only signed-zero readout comparison correction remains documented. Source-only continuation11ca4ac3 freezes all192 original chunks in `saved.py` and `SAVED_PROTOCOL.md`. These are reused observed panels, not fresh validation. No saved application has run; separate published source/resource and execution gates remain required.

## Fixed saved application completed

Raw45ab5489/shared32e3aa0f completes all192 chunks/196608 rows in10.535777677 main-scope CPU,385 archive reads. All unchanged-row readout checks pass. Heldout network agreement improves by53/75 net observations while task correctness loses56/17; all paired losses and context records are retained. `SUMMARY.md` reports both measures and hidden SSE. Independent full saved audit is pending; no predictor adoption or generic conclusion.
