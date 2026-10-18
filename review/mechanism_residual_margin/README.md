# T201: What the fitted reconstruction erases

## Question
Why does the fitted lookup solve the task more often than it reproduces the
original network? Which saved output-margin residuals accompany this gap?

## Method
Claim a21c54c9 follows the DRI's 02:10 redirect and the independently completed
T201 full fit (author d1f8dc54, reviewer 908f2a56). DESIGN.md freezes a distinct
saved-data discriminator. All three existing panels remain separate. The
source phase imports no author code and evaluates no model or fitted predictor.

## Status
Done for this bounded phase. Fourteen controls and all192 saved-data chunks are independently checked; see HANDOFF.md.
Reviewer recover_910 owns the complementary source gate. Execution requires a
separately published bounded protocol and reviewed implementation.

## Results
New raw results are in SUMMARY.md: the lookup erases2240/2183 original heldout errors and introduces449/470. Full saved verification aba9dd68 passes. The prior two heldouts have approximately 98% lookup
task accuracy but 95.3% agreement with the network. The source summary contains
the true task label, so task accuracy is not a reconstruction fidelity measure.
The original saved fit/results, failures and independent reviews are retained.

## Compute accounting
Current work is text/JSON/source reading and hashing. A later contraction must
charge lookup-hidden reconstruction, readout evaluation, residual terms,
classification, sorting, reductions and serialization as diagnostic work.
Inherited 196608 inputs, 217 fits and their forward passes are reused, not free
construction. No future cost/performance claim follows from a saved residual.

## Claims
[inferred] The real-arithmetic margin decomposition in DESIGN.md is an identity.
[speculative] A small subset of neurons may account for much of the discrepancy;
no neuron is selected and no correction is fitted in this protocol.
[inferred] Hidden residuals are oracle information about the original network,
not an available cheap estimator or a causal attribution to input features.

## Alternatives not ruled out
A few output-neuron directions may dominate, or residuals may be distributed.
The fixed all-16 signed/absolute tables discriminate without selecting neurons.
Errors may reflect fitted approximation or omitted input information; these
saved data alone do not separate the two. T138 already gives pointwise summary
non-sufficiency, but no useful population error lower bound.

## Failure modes
Floating matrix products differ from exact algebra. Preserve the observed
closure residual; do not interpret it as a certified error bound. First-index
ties, largest competing-class changes and zero denominators must remain explicit.
Bucket associations do not establish causation, noisy-weight transfer or an ARC
accuracy-versus-cost result. No historical percentage/provenance claim is added.

## Current checkpoint after controls

Fourteen controls completed in94ca93af and independently passed4459a768.
The original zero-call NumPy environment failure84989ffa remains in
ENVIRONMENT_FAILURE.md. Total recorded author control CPU is0.180262936;
independent saved review used0.035997058CPU and made no diagnostic calls.
The finite cancellation fixture is bookkeeping-only, not a network replay.

Historical source checkpoint: full saved-array source eff1656d initially awaited its separate source/resource gate.
It reconstructs the previously fitted lookup on all196608 original rows and
records every hidden residual, output margin, transition and fixed context.
These lookup/readout operations are new diagnostic arithmetic; no new recurrent
forward, fitting, Gaussian input, reference or mass operation is requested.
The partial scalar-expression ledger is not an installed native-operation cost.
That source checkpoint was superseded by completed raw88d44ff9 and independent full saved audit aba9dd68; SUMMARY.md is the current result.
