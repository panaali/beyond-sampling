# T201 residual-margin design review

## Question
Does the frozen saved-data decomposition distinguish task performance from network reconstruction without changing the fitted method?

## Method
Read-only review of DESIGN.md, the original lookup/readout source and twelve byte bindings. See source_gate.json for the exact scope and pins.

## Status
Source design passes. Implementation, controls and saved contraction remain separately gated.

## Results
The five transitions partition every observation. The fixed original competitor gives the stated sixteen-term identity in real arithmetic; first-index ties and other competing classes prevent replacing actual labels with a single positive margin. Floating closure discrepancy remains measured rather than asserted zero.

## Compute accounting
Text/JSON reads and SHA256 hashing only. No author imports, predictor evaluations, model calls, fitting, random draws or scientific controls.

## Claims
[inferred] The decomposition and partition are sound under the stated finite-vector and first-index conventions. [inferred] Residual attribution is descriptive oracle accounting, not a causal intervention or a cheap predictor of network error.

## Alternatives not ruled out
Distributed residuals, fitted approximation error and omitted input information remain possible. Existing data alone do not separate their causal roles.

## Failure modes
Tie exclusion, negative logits, cancellation, nonfinite intermediates and changing strongest competitors require explicit implementation controls. No implementation execution is cleared by this design review.
