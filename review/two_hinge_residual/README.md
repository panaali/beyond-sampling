# T233 — two-hinge residual prerequisite

## Question
Can opposite-slope ReLU pairs retain their amplitudes while expressing their
second independent gate as an explicitly controlled correction?

## Method
PROOF.md derives a pointwise nonnegative correction, its exact fresh-Gaussian
conditional probability and first moment, a raw-state-second-moment bound,
and downstream margin propagation. It assumes the full previous state is
conditioned on, without replacing its distribution by a Gaussian.

## Status
Proof-only claim c1343d8a published before writing. Independent written review
completed by recover_78:6c02b3aa/shared8106e746. Main theorem passes with execution_allowed=false. GAUGE_COROLLARY.md records the separately source-frozen diagonal invariance proof; durable review addendum pending. No implementation or numerical phase.

## Results
[inferred, independently reviewed] The correction is supported between two affine
thresholds and has a triangular Gaussian first moment. A density-cap bound is
quadratic in threshold separation. No target-neuron rarity or algorithmic gain
has been measured. No reproduction command exists because no experiment ran.

## Compute accounting
Zero model, geometry, mass, fit, RNG, target-array or numerical-library calls.
Writing and read-only source coordination only; no operation-count claim.

## Claims
All algebra is real-arithmetic and conditional on the stated nonzero opposite
slopes/fresh input independence. The signed surrogate is not a nonnegative
hidden state until its residual is restored. Classification accuracy requires
additional margin reasoning; no ARC success, cost-growth rescue or broad
negative conclusion is made.

## Alternatives not ruled out
Other anchors, multi-hinge grouping, reachable-domain bounds, exact residual
integration and rank-conditioned laws. Previous states need not be Gaussian.

## Failure modes
Same/zero slopes, rank-conditioned inputs, large downstream amplification,
unavailable state moments, naive CDF cancellation and expensive gate partition.
See PROOF.md for exact domain boundaries and proposed discriminators. Any later
controls or target use require a new scoped claim/gate and DRI priority.
