# T233 exact residual controls

## Question
Do the reviewed pointwise residual, conditional interval and gauge identities
survive an exact implementation, including anchor-zero boundary ownership?

## Method
`primitive.py` uses Fraction arithmetic only. `fixtures.json` fixes twelve
five-point cases; `controls.py` checks original ReLU differences, endpoint
polynomial integration and gauge invariance, plus eight malformed/domain guards.
`PROTOCOL.md` limits the one possible synthetic attempt; `protocol.json` binds
sources and original proofs. No target data or Gaussian evaluation is involved.

## Status
Source/control preparation. Independent source/resource review pending with
recover_34. No controls have run and no execution gate exists.

## Results
None yet. The mathematical identity was independently reviewed in original T233;
that does not certify this implementation. The irrational density maximum is
kept symbolic, not evaluated or counted as a computed Gaussian expectation.

## Compute accounting
Source preparation only. Zero primitive/control/model/MC/RNG/quadrature calls.
Future harness counts and main-scope CPU will be reported separately from whole
process limits and unpriced exact arithmetic. No performance benefit claimed.

## Claims
[inferred] Exact rational arithmetic avoids rounding in the tested identities.
[speculative] These identities may support a later useful conditional residual
method. No target rarity, cheap integration, accuracy or scaling claim follows.

## Alternatives not ruled out
Stable Gaussian expectation, non-Gaussian/truncated conditional laws, multiple
anchors, alternate orientation and source-cost-aware aggregation are untested.
The signed surrogate can be negative and must not be silently used as a valid
hidden state without restoring/bounding its residual.

## Failure modes
Endpoint ownership, slope-domain confusion, using area as expectation, dropping
mean terms in later state moments, and mistaking small expected margin changes
for small classification error. Any failures stay in immutable records.
