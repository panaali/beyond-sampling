# T55: fundamentals and successive falsifiable attacks

## Question
Can exact low-dimensional calculations be assembled into an efficient deductive
accuracy estimator for the small-model ladder, then the original H16,T10 problem?
Claude remains DRI. This is first-principles reasoning inspired by Feynman's
methods, not impersonation and not a repeat of the failed `sci_feynman` estimator.

## Method
The standing task proceeds in bounded cycles: derive the exact probability,
identify alternative explanations, try the cheapest falsifier, independently
review, and revise. Cycle 1 develops compact normalized-state transport and an
analytic two-input tail primitive. See `theory.md`.

Frozen initial numerical protocol (before implementation): implement weighted
spherical-polygon moments through degree 9; check coordinate octants, rotated
triangles, and thin triangles against an independent positive simplex integral.
Use deterministic cases, no trained checkpoints or model evaluations. Compare
ordinary and high-precision calculations if cancellation appears. Initial CPU
cap 120 seconds, one thread, memory below 2 GB.

Competing explanations and cheapest falsifiers:
1. The radial factor may remain dependent on the normalized state. Prove
   independence only for homogeneous direction summaries; demonstrate that
   absolute-magnitude conditioning invalidates the argument.
2. A plausible boundary recurrence could have wrong orientation or normalization.
   Test full hemisphere assembled from octants and positive simplex integration.
3. An exact expression may be numerically unusable on thin regions. Include
   near-equatorial and nearly coincident vertices, without clipping negatives to
   zero or hiding failed cases.

## Status
Running. Theory derived; T47 agent independently checked the density and Green
identity algebra. No full estimator or solved claim. Runtime recovered after an
outage; the assigned worktree was recreated from snapshot 72edef0. The requested
small-ladder CHECKLIST was absent from that recovered snapshot.

## Results
No numerical results yet. This protocol is committed before numerical tests.

## Compute accounting
Cycle 1 theory used no model forward passes, random inputs, or experiments.
Numerical moment validation will report CPU time and quadrature evaluations.
Full geometric partitioning and outer transport costs are not yet measured.

## Claims
- [inferred] H2 accuracy admits a target-specific Markov formulation with three
  continuous normalized coordinates and a capped exceedance count, independent
  of sequence length. Low state dimension does not imply a cheap representation.
- [inferred] Exactly two remaining Gaussian inputs can be integrated over weighted
  spherical polygons; the density and recurrence have independent algebra review.
- [speculative] Combining these primitives could be competitive with sampling.

## Alternatives not ruled out
Exact geometry may still cost more than sampling. Compact-state laws may acquire
too many singular pieces. Reset decompositions may only help special checkpoints.
No generalization or computational advantage follows from the current derivation.

## Failure modes
Absolute-magnitude conditioning breaks radial independence. Positive-volume logit
ties require first-index argmax; boundary measure-zero arguments do not remove
them. Floating-point cancellation can invalidate tiny-region probabilities.
Random outer prefixes produce a sampling baseline, not a deductive solution.

## Continuity
Keep this task active across successive cycles. Root owns backlog updates,
integration and communication with Claude; only this folder is agent-owned.
