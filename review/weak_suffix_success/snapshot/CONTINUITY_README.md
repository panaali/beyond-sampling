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
Bounded trained phase complete after independent source/theorem review.
Independent exact coefficient and branch reconstruction review d8bc064
passes. Result classification remains with DRI.

## Results
Reproduce from this folder using the project interpreter: `python controls.py`
and `python pilot.py`, with one thread/review lock/memory cap.

All5 exact toy controls behave as frozen. The trained model satisfies every
required slope condition: both gates and all9 nonconstant target-margin
support cases are nonzero exact rationals. Thus this conditional value
function is Lipschitz in hidden state under the stated law. A jump at an
axis/zero-support boundary is not the explanation for these particular
interpolation errors. This conclusion is model/label-specific, not a claim
about arbitrary recurrent models or all support degeneracies.

The exact coefficient G has numerical value about2.96267e8; at u=1 the
displayed G/min(L,U) is1.86736e9. This extremely loose bound gives no useful
mesh error or accuracy certificate. Nearzero exact slopes are accepted as
nonzero for continuity and retain their full large cost in the bound.

For x0=-1/4, both actual need1 branches agree to roundoff; side00 alone has
error-.00698826 and is not a physically consistent remaining need here.
For x0=+1/4, both actual need1 branches are biased low:

| Rank side of x,y | Exact pair | Old interpolation | Half error contributing to need1 |
| --- | ---: | ---: | ---: |
| below,above | 1.0000000000 | .8767810617 | -.0616094692 |
| above,below | .1922328311 | .0440280048 | -.0741024132 |

Their errors add to-.1357118823; there is no beneficial cancellation. The
single-branch explanation is unsupported by this discriminator. All four
rank-side arrays and their exact reconstruction of the saved T127 need
values remain in pilot.json. Complete pilot CPU .02031. No full integration,
larger grid, population-error estimate or computational victory follows.

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
