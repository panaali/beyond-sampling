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
Protocol commit: `1d68fde`, before numerical implementation and tests.

Reproduce the first primitive check:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python m16/approaches/feynman_fundamentals/check_moments.py
```

`moment_results.json` records nine deterministic triangles, degrees 0 through 9.
Maximum absolute disagreement is 6.89e-15 against independent positive simplex
quadrature and 1.12e-16 against the 100-digit version of the same recurrence.
Summing four octants gives hemisphere probability one within 1.12e-16 for k=1..10.

The thin-equatorial tests falsify unrestricted double-precision reliability:
one degree-9 moment is approximately 1.65e-18 in double precision but 7.56e-70
at high precision and under the positive integral. A geometric upper-bound check
rejects this result and ten other thin-case degree choices. It is a detection
guard, not an error certificate. No probability is silently clipped.

Cycle 2 (`1e5b93c` protocol) now has `tail.py`: it takes a normalized prefix
hidden state plus its top-two values/indices, enumerates the final two steps,
and analytically integrates their rank/readout polygons. Reproduce validation:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python m16/approaches/feynman_fundamentals/check_tail.py
```

`tail_results.json` contains seven fixed synthetic conditional cases (H1T3 and
H2T3/T5/T10). The four H1 cases give the analytic 1/3 to floating precision.
All seven agree with separate 65,536-draw direct validation within 1.83 standard
errors. Multiplying input and output weights each by 1e-16 changes the answers
by at most 1.39e-17. Gate-partition mass differs from one by at most 2.23e-16.
These are conditional probabilities for fixed prefix directions, not solved
long-sequence accuracies. Independent implementation review is pending.

## Compute accounting
Cycle 1 uses zero model forward passes and zero random inputs. Each complete
check uses 184,320 simplex quadrature nodes; the two runs used 368,640 nodes and
approximately 0.481 CPU seconds in total. The saved final run took 0.284 CPU
seconds, including high-precision checks. Full geometric partitioning and outer
transport costs are not yet measured; no efficiency advantage is claimed.

Cycle 2 used 0.241 CPU seconds, 458,752 full model inputs for validation, and
14 deterministic prefix calculations totaling 32 recurrent steps. The analytic
tail calls use no model-input sampling. Normal and rescaled calls together used
1,678 polygon clips, 136 affine products, and 222 weighted-moment calls. No
high-precision fallback occurred; these selected cases did not trip the guard.
The counted operations are not yet converted to an audited FLOP/FPE budget.

## Claims
- [inferred] H2 accuracy admits a target-specific Markov formulation with three
  continuous normalized coordinates and a capped exceedance count, independent
  of sequence length. Low state dimension does not imply a cheap representation.
- [inferred] Exactly two remaining Gaussian inputs can be integrated over weighted
  spherical polygons; the density and recurrence have independent algebra review.
- [measured] The bounded primitive check agrees with independent positive
  quadrature to 6.89e-15 absolute error, but unrestricted relative accuracy fails
  on thin near-equatorial regions.
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

## Cycle 2 protocol (frozen before implementation)
Build a two-final-input conditional probability from normalized prefix hidden
state, its largest two inputs and their indices. Enumerate only the two tail
steps, then rank/readout polygons; integrate with the weighted sphere primitive.
No long-prefix estimator or trained-model performance claim in this cycle.

Alternatives: (1) correct moments but incorrect rank/first-index tie handling;
test H1 zero-recurrence/all-tied outputs and deterministic direct classifications;
(2) truncated geometry or normalization produces biased probabilities; compare
conditional outputs against independent random-direction validation, include
positive rescaling and partition-mass checks; (3) work per tail outweighs variance
reduction, so record geometric operations and CPU without asserting an FPE gain.

Use fixed synthetic H1/H2 cases, T3 and T5/T10 prefixes, at most 65,536 independent
validation draws per case, at most 12 cases initially. Single thread, 120 CPU
seconds total initial experiments across both cycles. Random draws are validation
only; the primitive itself is analytic. Preserve all failure cases. High-precision
fallbacks must be counted; no silent clipping of small negative probabilities.
