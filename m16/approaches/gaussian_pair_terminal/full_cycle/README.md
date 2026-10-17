# T118: bounded full transport using Gaussian pair terminals

## Question
Does removing final-stage interpolation for every early target improve full
accuracy on a fixed small grid, once rank/event measure and cost are retained?

## Method
H2T4 seed0, h3 only, input[-6,6], target[-4,4]. Both arms use the established
exact penultimate-target terminal at stage2. Baseline labels0/1 use the old
stage3 bilinear terminal interpolation; new labels0/1 start at the exact
Gaussian pair terminal on the same stage2 nodes. Earlier recurrence,
first-index rules, grids and rank conditioning are identical. Label3 must
pass the existing componentwise readout-domination exclusion. No selected
mix of component truths or measured estimates enters either estimator.

## Complete target-event topology
A fixed pair's correctness is a union of convex polygons in(x,y). Adding
rank cuts x=u,y=u can change the clipped polygon's combinatorial type only
when either line crosses an old vertex (u equals its x or y coordinate), or
when their intersection(u,u) crosses an old edge. Thus include all original
vertical trapezoid vertex coordinates and all edge/diagonal intersections.
Artificial verticalization edges add harmless extra events. A boundary
identically on the diagonal adds no isolated root. Union these events over
all stage2 states/labels with the complete previous grid/penultimate knots.
Use identical union panels in both arms, q-space GL2 andGL4. This real-
arithmetic topology argument is not a floating predicate or quadrature bound.

## Frozen alternatives and falsifiers
1. Final interpolation dominates: pair replacement improves component and
   full error at this fixed grid. Report all components, even if cancellation
   makes total error behave differently. No component winner selection.
2. Earlier interpolation dominates: total/component improvements may fail;
   retain the result, do not retire the pair primitive or transport family.
3. Outer integration or rank topology is wrong: original ties/constants,
   hidden scaling and exact Beta mass controls first; both GL2/4 changes
   must be<1e-7 and<10% of any claimed full-error improvement.
4. Query cost dominates: construct topology first; cap2000 panels and50000
   pair queries across both new-arm rule orders, excluding separately logged
   controls. If either cap fails, preserve setup and stop before full runs.

Numerical pilot only after independent topology/source review. Exactly four
full estimates: baseline/new xGL2/4; shared h3 and union topology. Combined
internal CPU cap100seconds, one thread/2GB/primarylock. No h9, larger order,
noise/model expansion, native speedups, adaptive panel selection or full
challenge claim. Inconclusive precision is recorded, never silently relaxed.

## Status
Claim b981947 pushed before implementation. Protocol frozen now.

## Results
Pending. Fixed full/per-label cone references are diagnostics only.

## Compute accounting
Full cold grid, old/new geometry, original halfspaces, all line intersections,
sorting, Gaussian quad callbacks/errors, sparse kernels and recurrence CPU.
New-arm queries integrate only rank masks actually used by need0/1 eventually,
but this first version preserves the verified primitive including side11;
its redundant work is charged. Pair error estimates propagate through positive
operators as numerical diagnostics only. Fine-grained counters are partial;
no certified total floating error or portable-FLOP/sampling claim.

## Claims
[inferred] The real-arithmetic replacement removes one interpolation stage.
[speculative] The complete estimator improves accuracy/cost at fixed h3.

## Alternatives not ruled out
Earlier-state representations, exact state strata, batched analytic Gaussian
polygon queries, rigorous mass enclosures and different tensor bases remain.

## Failure modes
Unreachable grid-state extension, missed topology, first-index ties, small
rank normalizers, nonconverged local quad, finite precision, cancellation,
and local accuracy improvements overwhelmed by full query count.

## Setup checkpoint and preserved constant-control failure
Complete h3 union topology has178 panels and predicts19224 pair queries for
both new-arm orders, within the frozen caps. Setup costs .0237CPU. T55
independently accepts the vertex/diagonal event argument in real arithmetic.
Before trained evaluation, the constant-label control finds a QUADPACK
failure on a2-ulp x interval near-.5472948572881893, rank11,u=-3.8039629.
Its positive returned mass is3.81e-17, but adaptive subdivision cannot resolve
such an interval and the operator correctly raises. Exact bounds, model
state and message are preserved in initial_controls_failure.json. No trained
full estimator has run; this is a numerical implementation limit.

Freeze a local repair: only when interval width<=8 machine eps times
max(1,|endpoints|), use a positive midpoint integral and a real-arithmetic
range enclosure built from extrema of Gaussian density and the affine-bound
CDF gap on that interval. Retain its error diagnostic; it is not directed
floating arithmetic. Check the recorded fixture at80 digits and both signs,
repeat existing primitive and full constants before the trained pilot.
Do not suppress QUADPACK failures on wider intervals or omit unused rank11
just to hide this defect. All counters include the fallback.
