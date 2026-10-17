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
long-sequence accuracies. T47's bounded independent implementation review passed;
see `INDEPENDENT_REVIEW.md` and its reproducible `independent_review.py`.

`python m16/approaches/feynman_fundamentals/check_rayleigh.py` adds four closed-form
T4 constant-readout checks, derived by Rayleigh integration by parts in
`theory.md`. They agree within 1.12e-16; see `rayleigh_check.json`. No sampled or
quadrature reference is used for these four tests.

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
The additional four Rayleigh checks used 0.0613 CPU seconds, four deterministic
prefixes/eight recurrent steps, 208 clips, 32 affine products and 29 moment calls;
zero full model-input evaluations.
The independent reviewer additionally used 0.1345 CPU seconds, six 1-D positive
integrals and 262,144 direct conditional validation inputs. These are separate
from the author totals. No shared leaderboard or trained checkpoint was run.

Cycle3 positive fallback: `positive.py` supplies a positive integral with an
analytic quadrature-only error bound. The T48 agent independently checked its
proof in `theory.md`. `tail.py` now automatically invokes the fallback after a
recurrence guard failure and records its nodes/count/error bound in statistics.
The old `moments.tail_mass` still exposes original failures for regression checks.

Reproduce under the shared compute lock:

```
flock /workspace/work/codex-compute.lock env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python m16/approaches/feynman_fundamentals/check_positive.py
```

Seventy configurations (seven polygons, five k values, two tolerances) used
91,472 nodes, maximum order96, and0.088 CPU seconds. Maximum disagreement with
100-digit recurrence was1.48e-15. Twenty-four cases exceed their extremely small
quadrature-only bounds because floating-point errors are separate; all are
preserved in `positive_results.json`. No full numerical certification is claimed.
The T51 agent supplied `trained_moment_failure.json`: its k2 guard failure is
resolved using16 positive nodes, agreeing with high precision to1.17e-27 absolute.
A separate direct regression call used approximately0.038 process wall seconds;
neither check used model forward passes or random inputs.

Cycle4 adapter: `target_tail.target_probability(model,a,b,K,j)` provides the
two-fresh-input terminal on T50's normalized target/count state. It inserts the
preselected target in the correct chronological location and handles count2
absorption, axes/origin, b=0 and first-index ties. Some arbitrary grid states
are unreachable; the callable defines an operator extension there.

Reproduce `check_target.py` with the same flock/thread prefix above. Twelve
fixed cases used786,432 direct validation draws in65,536 chunks and0.332 CPU
seconds. Maximum discrepancy was2.26 SE; input/output scaling by1e-16 changed
the kernel at most1.12e-16. For synthetic T3, summing all three target-specific
passes gives .32421191293570956 versus .3242119129357096 from the separately
partitioned top-two tail (difference5.56e-17). No positive fallback occurred.
The kernel and its validation are not a full estimator or an efficiency result.

Statistics now explicitly use `positive_quadrature_fallbacks`; `positive_fallbacks`
is a compatibility alias. `high_precision_fallbacks` remains0 and does not count
quadrature. Additional fields expose quadrature-node counts, quadrature-only
error bounds, newly generated Legendre rules and their orders' node counts.
`method_class` states `analytic_regions_with_numerical_quadrature_fallback`.
The numerical fallback must not be described as exact arithmetic or as covered
by a floating-point certificate.

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

## Cycle 3 protocol (frozen before implementation)
Build a positive simplex-quadrature fallback for weighted spherical moments with
an explicit analytic quadrature-error bound. It must expose the bound and node
count, rather than accepting agreement between two numerical approximations as
certification. The bound will be conditional on supplied geometry and real
arithmetic; floating-point clipping/rounding remains a separate qualification.

Two alternatives: (1) Duffy degree counting or the positive-weight error factor
is wrong; obtain an independent algebra review and compare against exact octant
moments. (2) The bound is correct but too expensive; test ordinary and thin
triangles from cycle1 at fixed absolute tolerances 1e-8 and1e-12, recording nodes
and CPU. The cheap falsifier is a bound smaller than observed high-precision
error, or inability to reach1e-12 within128 nodes per dimension.

All numerical runs acquire `/workspace/work/codex-compute.lock`, use one BLAS
thread and remain below2GB RSS. Cycle budget120CPU seconds. No trained T4 outer
integration: its assigned owners retain that work. Root receives checkpoint
commits and the standing investigation continues after each bounded cycle.

## Cycle 4 protocol (frozen before implementation)
Provide T50's backward collocation owner an exact-formula terminal kernel on
(a=h/R,b=u/R,K,j), with two fresh non-target inputs remaining. If j is already
processed, perform two random steps; if j is penultimate, insert bR between them;
if j is last, insert it after them. Radius dimension is always T-2. K=1 requires
both fresh inputs below u; K=0 requires exactly one above. Positive-volume
logit ties use first-index argmax, including dead states. Grid construction and
interpolation remain T50-owned, not part of this cycle.

Competing explanations: (1) wrong target insertion order/radius normalization;
test every target position of synthetic T3 against the independent top-two tail
and direct conditional simulation. (2) count boundaries or zero/axis states
silently lose mass; include a=0, b=0, counts0/1/2, identical output rows and
positive input/output rescaling. Up to12 fixed cases with65536 validation draws
each, one bounded locked run initially. Keep every failure. Analytic kernel
calls use no sampled model inputs; the check's random draws are validation only.

## Cycle 5 protocol (frozen before implementation)
Supply T48 with a stable vectorized formula for the inverse-sixth-power norm
integral along a segment whose endpoints share an orthant. This analytically
eliminates the last Duffy coordinate in T6 spherical-simplex integration.
Alternative 1: the angular expansion has a missing scale or Jacobian factor;
compare independent high-precision integration on equal, orthogonal and unequal
endpoint norms. Alternative 2: the formula cancels near parallel endpoints;
use a positive coefficient series with a geometric remainder bound, test
cosines near one and verify positive scaling. Preserve all discrepancies.

Initial tests use fixed endpoint pairs, no model samples, one locked numeric
run with at most120 CPU seconds and at most2GB RSS. Report scalar endpoint
evaluations, coefficient-series terms and actual CPU. The series bound excludes
floating arithmetic and the closed-form coefficient branch; do not label it a
full numerical certificate. T48 retains trained T6 integration ownership.

Cycle5 result: `segment.inverse_six_segment(A,B,diagnostics=False)` broadcasts
endpoint arrays of shape (...,dimension), returning one analytic integral per
pair. With diagnostics enabled it returns (values,stats); the stats include an
array of series-truncation bounds, NaN for the closed-form branch. Independent
80-digit direct quadrature of28 fixed pairs (norm ratios through10000, equal
and perpendicular directions, branch threshold, and nearly parallel inputs)
gave maximum relative error3.411e-15. Scale checks gave5.996e-15 and endpoint
swaps were identical. This is a validated primitive, not a solved T6 model.

Reproduce with single-thread environment variables and
`flock /workspace/work/codex-compute.lock python m16/approaches/feynman_fundamentals/check_segment.py`.
Measured author CPU1.5765 seconds for the entire check;84 analytic endpoint
evaluations including rescaling and swapping, no model passes or sampled
inputs. The main28 evaluations used56 norms,28 dot products,16 positive-series
evaluations of28 terms and12 closed-form evaluations. Independent numerical
reference quadrature is validation overhead, not part of the estimator.
Results and every tested endpoint are saved in `segment_results.json`.
