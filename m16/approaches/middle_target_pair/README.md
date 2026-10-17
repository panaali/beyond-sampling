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
Protocol frozen before implementation.

## Results
Pending. Deductive finite local diagnostic, no model-input sampling.

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
