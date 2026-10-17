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

## Wider narrow-panel failure and distinct numerical rescue
The explicit2-ulp fallback matches80-digit positive integrals (three original,
reflected and signed fixtures) within9.8e-33, with a deliberately conservative
uncertified floating allowance. Repeating full constants exposes another
QUADPACK domain-conditioning failure on a128-ulp interval of width5.64e-14
near-2.18918; this is outside the frozen tiny fallback threshold and remains
an explicit failure in second_controls_failure.json. No threshold is widened.

Freeze the algebraically exact affine change x=a+(b-a)t for every remaining
adaptive integral, integrating t on[0,1] and scaling both value/error by width.
This avoids asking the adaptive driver to subdivide an almost indistinguishable
physical-x interval. Divide its absolute tolerance by width to preserve the
same requested raw-probability tolerance. Keep the explicit2-ulp fallback,
all guards, geometry and every wider nonconvergence rejection. Re-run both
known fixtures and prior controls; no full trained pilot before they pass.

## Unit-coordinate repair verification
The128-ulp fixture now agrees with80-digit integration within2.65e-31;
its returned numerical error estimate is1.14e-29. Existing36 toy panels
still partition/agree by axis within4.45e-16, direct scalar checks1.06e-12,
and scaling2.23e-16. The frozen18 trained primitive queries are repeated
because the integration kernel materially changed: axes agree4.45e-16,
.0680CPU, recorded as unit_pilot.json; original pilot.json is preserved.
Full constant/duplicate baseline/new controls pass rawmass/T within2.78e-17,
and six full conditional scaling probes agree1.12e-16, in5.256CPU.
Both original failures remain recorded; this repair changes integration
coordinates without dropping geometry or relaxing convergence gates.
The tiny fallback range uses separate extrema of upper/lower affine bounds,
not extrema of the CDF gap at matching endpoints. T55 independently confirms
that distinction; explicit floating padding is heuristic, not certification.
Source review is pending before the frozen four-estimate trained pilot.

## Independent source review before trained evaluation
T55 checks stage2 terminal chronology, remaining-count mixture through t1/t0,
complete vertex/diagonal knots and the numerical-rescue scaling. Its suggested
explicit early-label guard is adopted: pair_label is dispatched only for
label<2. The inherited prepare already rejects any model where target-last
label3 is possible; this frozen base passes that exclusion. No support for
a general target-last model is silently inferred. Local quad/roundoff errors
remain diagnostics, not certificates. The frozen four-estimate pilot proceeds.

## Frozen full result: strong representation gain, precision gate fails
All four estimates complete in11.571CPU seconds, below the frozen cap.

| Arm | GL4 accuracy | Error | Component L2 | GL2/4 change | GL4 CPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline with same penultimate cache | .256589235103 | -.051141507921 | .0446998964 | 7.66826e-6 | 1.3404 |
| exact early-label Gaussian pairs | .301712330584 | -.006018412440 | .00664234763 | 7.20229e-6 | 6.1644 |

The pair arm reduces absolute full error8.50x and component L2 by6.73x,
while CPU grows4.60x. Its measured error-squared-times-CPU is15.70x smaller
on this single fixed comparison; this is not a sampling comparison or a
portable-cost claim. Both arms fail the strict1e-7 outer-stability gate.
The GL2/4 shifts are much smaller than the .04512 full-probability change,
but no convergence threshold is relaxed and no larger-grid expansion follows.

GL4 per-label errors baseline:
[-.044223807064,-.000425013252,-.006492687605,0].
Pair arm:
[-.000725494554,+.001199769719,-.006492687605,0].
Label0 improves strongly; label1 worsens; label2 is identical and dominates
the new error. All signed components are retained rather than selecting an
estimator from them. The remaining label2 GL2/4 change7.70e-6 accounts for
most of the shared outer drift. Floating geometry/quad diagnostics remain
uncertified, and this off-regime H2T4 result does not solve the full challenge.

## Bounded handoff
Implementation defects, local representation error and outer precision were
separated rather than collapsed into a single negative. Independent review
checks the measure, terminal, rank topology and recurrence; final arithmetic
summary review requested. Broad transport remains open. Untested alternatives
are earlier-state/prefix representation for the penultimate target, event-local
outer error localization, and faster positive Gaussian polygon integration.
No larger h, higher global quadrature order, model/noise sweep or full-cost
sampling claim is justified under the failed frozen precision gate. The
next work item must state a distinct discriminator and preserve these results.
