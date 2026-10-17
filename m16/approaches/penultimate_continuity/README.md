# T132: continuity and rank-side localization of penultimate error

## Question
T127 found a large h3 interpolation discrepancy at a nonzero reachable state.
Is a discontinuity at hidden-support boundaries a plausible explanation,
or does a sufficient continuity theorem apply to this model? Which of the
two actual remaining-rank branches contributes the discrepancy?

## Method
Claim b3cf514 pushed before implementation. Fix chronology h -> x -> u -> y,
target label T-2, independent original Gaussians truncated to [-6,6], fixed u.
For initial-state distance d in infinity norm, final preactivations differ
by at most M³d, M=||Wh||infinity. Final gate flips lie in bands around the
reference preactivation roots in y, of total length at most
`2*M³*d*sum_i 1/abs(Wi_i)`, provided all Wi_i are nonzero.

If the two final supports agree and are nonempty S, correct-event ownership
can change only at a target-versus-competitor margin. Its y slope is
`s=(Wo[label]-Wo[j]) dot (Wi*1_S)` and its perturbation is at most
`||Wo[label]-Wo[j]||1*M³*d`. If the restricted readout difference is zero,
the tie is permanent and uses first-index ownership. Otherwise require
exact rational s!=0. The margin-changing set has y length bounded by
`2*||deltaWo||1*M³*d/abs(s)`. Union over all3 nonempty H2 supports and all
competitors. On empty support all outputs tie identically and label0 owns
them; a change leaving that support has already been counted as a gate flip.

Let G be the sum of these exact rational coefficients. Conditional y density
is at most `1/min(L,U)`, using phi<1, L=Phi(u)-Phi(-6), U=Phi(6)-Phi(u).
Thus each rank-side correctness probability, and both need mixtures, differs
by at most `min(1,G*d/min(L,U))`, uniformly over x. This is a deliberately
loose real-arithmetic Lipschitz bound. Numerical CDF displays are not outward
certificates. A zero required slope produces **unknown**, not discontinuity.
When M=0 the continuation is state-independent and G=0 without slope tests.

Freeze exact controls: memoryless state independence; constant readout ties;
a nondegenerate model; and a nonzero restricted readout difference with
zero y slope that must be reported unknown. Independent proof/source review
before the trained certificate and branch diagnostic.

For exactly the previous two x0=+-1/4,u=1 probes, report the old stage2
interpolation's complete2x2 rank-side conditional array: low and high matrix
products against the penultimate two-column terminal. Compare all four with
the existing exact pair arrays. Need1 discrepancy is half the sum of errors
in sides01 and10; report each, including any cancellation. No new target,
state scale, hidden grid, rule order, model or noise panel.

## Alternatives and falsifiers
1. A support-boundary discontinuity matters: the sufficient theorem may
   fail; failure of this criterion alone cannot establish discontinuity.
2. Continuous but sharply varying values cause error: exact slope tests
   pass while one rank branch has a large old/new discrepancy.
3. Rank mixing or measure is wrong: four-side arrays must reproduce saved
   need values and memoryless analytic controls before interpretation.
4. Cost defeats replacement: charge all fixed primitive/comparator setup;
   no full-estimator or portable-cost conclusion from local comparisons.

## Status
Protocol frozen. No trained computation yet.

## Results
Pending. Reproducible theorem/control and branch scripts will be retained.

## Compute accounting
One thread,2GB, review lock, <=20CPU per bounded phase. Exact rational sums,
products, divisions and slope checks plus full old/new geometry/query work.
No sampling, prior panel expansion or hidden uncharged full solve.

## Claims
[inferred] The stated sufficient nondegeneracy condition implies Lipschitz
continuity of the conditional correctness value in the initial hidden state.
[speculative] The trained discrepancy is concentrated in one rank branch.

## Alternatives not ruled out
Adaptive state representations, exact lower-dimensional support handling,
different value bases, geometry batching, full population integration and
rigorous floating-probability enclosures.

## Failure modes
Omitted gate changes, treating nonconstant zero-slope margins as permanent
ties, using a floating nearzero slope as proof, wrong rank normalization,
or mistaking a very large Lipschitz constant for a useful error certificate.
