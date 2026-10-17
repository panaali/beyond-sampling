# T127: two-Gaussian terminal with the target in the middle

## Question
Does stage2 hidden interpolation explain the remaining penultimate-target
error in T118? Its label2 contribution is low by .00649269 at h3, even after
the early-label pair replacement. A fixed-state conditional discriminator
can isolate this interpolation without another outer quadrature sweep.

## Method
Claim2634e17 pushed first. For H2 and label T-2, use chronology
`h -> random x -> fixed target u -> random y -> readout`. Split the x axis
at the first transition gate roots and at every middle-target gate root;
then each final gate/readout region is an affine polygon in (x,y). Reuse
the verified positive Gaussian polygon integrator from T117/T118, keeping
all absolute offsets, first-index ties, input truncation and strict guards.
The geometry now depends on u and must be rebuilt/charged for each query.

With low/high masses L=Phi(u)-Phi(-6), U=Phi(6)-Phi(u), rank-side polygon
integrals Pab use the original Gaussian density. Remaining-rank values are
V0=P00/L² and V1=(P01+P10)/(2LU). Target insertion is deterministic and
does not consume a remaining-above count. The fixed state is allowed to
depend on earlier inputs; conditioning on it and u is sufficient.

Freeze toy controls: constant, duplicate, zero gates, Wh0 and ordinary
nonzero hidden states; swapped integration axes, hidden rescaling, and
different direct scalar recursion. The primitive must reject a target
query different from its construction value. Independent source and law
review precedes the trained panel.

Trained diagnostic: existing H2T4 seed0 checkpoint, h3 representation,
prefix x0 in {-1,0,1}, target u in {-1,0,1}, total9 fixed queries. Report
both conditional need values, and separately the6 untied physically
rank-consistent values: need0 when x0>u, need1 when x0<u. Matching x0=u is
a zero-probability input tie and not counted in that subset. Check complete
[-6,6] image support against the old hidden grid; no clamping/extrapolation.
Compare exact pair values against one old stage1 conditional transition
acting on the existing exact penultimate stage2 terminal. Report every
signed discrepancy, setup/query work and CPU, no favorable selection.

## Alternatives and falsifiers
1. Hidden interpolation matters: local exact pair differs materially from
   the old operator on untied reachable states. This is not population bias.
2. Outer integration or earlier interpolation dominates: small local
   discrepancies leave those explanations open; no family is retired.
3. Chronology/rank/tie error creates discrepancies: require analytic,
   partition, swapped-axis and independent scalar controls first.
4. Query geometry is too expensive: charge all construction and callbacks;
   no full transport or larger grid follows from local accuracy alone.

## Status
Bounded phase complete through dfacb39. Independent source/primitive review
70f2b21 and nonzero-prefix scope/result review39f7deb pass. DRI classification
pending; next task T132 isolates continuity and rank branches.

## Results
Results below. Deductive finite local diagnostic, no model-input sampling.
Reproduce in this folder with the project interpreter: `python controls.py`,
`python check_interpolation.py`, `python pilot.py`, `python nonzero_prefix.py`.
Use one thread, the review lock and the stated memory cap.

## Primitive controls
Thirty toy panels pass partition to4.45e-16, axes to3.34e-16, exact binary
rescaling with zero discrepancy and memoryless closed-form checks. Initial
direct checks used only label2, which was dominated in the toy; that weak
control is preserved in initial_controls.json and expanded to all4 labels.
The24 independent scalar comparisons differ by at most3.61e-11. One scalar
reference reports QUADPACK roundoff/nonconvergence; its message and error
estimate remain explicit in controls.json. It is a numerical diagnostic,
not a converged or certified reference. The analytic Wh0 and partition/tie
controls do not depend on this adaptive scalar reference. Full control CPU
approximately1.51 seconds on the final recorded run. No trained query before
independent source review.

## Nine-probe result and frozen nonzero-prefix check
Independent source/law/comparator review70f2b21 passes; its separate
identity-memory nontrivial calculation agrees to1.12e-16. The old comparator
also passes Wh0 analytic controls to1.12e-16 in .0213CPU. The frozen9 trained
probes complete in .0441CPU. An initial serialization error (builtin abs on
a list) and its unchanged-query rerun are preserved explicitly.

Swapped axes agree to1.12e-16. All source and complete image supports lie
inside the old grids. Of the6 untied rank-consistent values,5 agree to1.12e-16;
the exception x0=0,u=1,need1 gives exact pair .59696485871 versus old
.46132425722, difference-.13564060149. Exact zero is a special prefix and
does not by itself establish a positive-mass interpolation error. Other
conditional need flags remain extension diagnostics, never population error.

Scope update c50b34f pushed before the follow-up: evaluate exactly two
nonzero prefixes x0=-1/4,+1/4 with u=1. Both need1 are rank-consistent.
Compare old/new values, swapped axes and the independent direct scalar
recursion. No additional offsets, tuning or outer integration. Preserve any
scalar nonconvergence explicitly and do not promote its error estimate to
a certificate. This discriminates a zero-state artifact from discrepancies
away from zero; it still cannot quantify full population bias.

## Nonzero-prefix result
The discrepancy persists at x0=+1/4,u=1,need1: exact pair .596116415572,
old interpolation .460404533263, difference-.135711882309. At x0=-1/4,
the rank-consistent need1 value .694035425453 agrees to1.12e-16. Report
both sides; the inactive need0 values differ but are not actual rank states.
Swapped axes agree5.56e-17; separate scalar recursion agrees2.28e-11 with
no warnings on either nonzero probe. Full two-probe comparison including
scalar references costs .250CPU. The large difference therefore occurs
away from exactly zero prefix, but its population-weighted size is unknown.

The primitive provides a valid fixed-state alternative, not yet a full
estimator or portable-compute gain. Before any full integration, the next
question is whether the dominant local error comes from a discontinuity
between zero/axis/interior hidden-state supports or ordinary curvature.
That requires a separately frozen discriminator, not an unbounded mesh sweep.

## Compute accounting
One thread, 2GB, review lock and <=20CPU per bounded control/panel phase.
Charge geometry, intersections, density calls, callbacks, scalar controls,
old-grid setup and query operators separately. No portable FLOP claim.

## Claims
[inferred] The polygon representation removes the intervening hidden grid.
[speculative] This materially improves the remaining penultimate component.

## Alternatives not ruled out
Full population-weighted interpolation error, outer convergence, earlier
state representations, rigorous floating probability bounds and batching.

## Failure modes
Wrong insertion chronology, missing middle gate root, accidental homogeneous
zero-state assumption, unstable affine predicates, rank underflow, tied
prefixes misreported as typical support, or costs overwhelming local gains.
