# Exact third-step means

## Question
Can preserving the exact first-two-step angular/radial structure produce the
third hidden-state mean without Gaussian closure?

## Method
First analytic gate compares the reviewed Rayleigh/Gaussian scalar identity
and a distinct Gaussian facet first-moment identity. See primitive.py and
PROTOCOL.md. Exact dyadic normal algebra handles multiplicity and degeneracies;
fixed-precision MP handles transcendental values, with independent references.

## Status
Running under claim2666600d. Initial35 analytic/domain controls complete
a24b97ba, independently88f212b6; six analytic network-assembly controls now pass under claim d7c0edab,
independently7ad8ea7a. One pinned trained H16T4seed3 phase now also passes
geometry4142246c and aggregatefb944bce. See trained/HANDOFF.md.

## Results
All35 initial controls pass; maximum relative independent-reference difference
3.840e-120. See HANDOFF.md for scope, cost and reproduction. One historical model now has all544 exact-real facet contributions checked
against independent angular references. Mean sample-relative RMS0.00262984;
no covariance or final-accuracy inference.

## Compute accounting
Named exact arithmetic and reduced operand bits, MP/reference CPU separately.
No model forwards, random inputs, truth regeneration or geometry panel calls.
Complete runtime, covariance and downstream costs remain open.

## Claims
[inferred] Both formulas are exact on the stated real domains.
[speculative] A finite facet formula may avoid angular quadrature in h3 means.

## Alternatives not ruled out
Signed cancellation, angular-reference limitations and source sector rounding.
Better means need not improve covariance, step4 closure or final accuracy.

## Failure modes
Wrong duplicate/redundant facet handling, finite MP cancellation, degenerate
sector conventions, unpriced setup. Prior T69 random-width and trained errors
are preserved; no method-family retirement or benchmark win follows.
