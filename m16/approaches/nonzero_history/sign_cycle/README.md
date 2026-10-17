# T76 first-sign conditional history mixture

## Question
Does retaining the sign of the first omitted input repair the deterministic
mean-history failure? Claim9ea9d11 was pushed before implementation. Earlier
unconditional-mean code/results remain unchanged; this is a distinct mixture.

## Method and frozen variants
For each sign s of X0, probability1/2, compute the exact conditional two-step
mean (hypot(c_s,Wi)+c_s)/sqrt(2*pi), c_s=Wh relu(s Wi). For any further prefix
steps use the same frozen deterministic Gaussian-mean closure as before.
T4's conditional means are exact; T5's third-step closure is approximate.
Run two variants: correct sign-to-mean association, and shuffled association
(mu_plus used for the negative rank law, and vice versa). The shuffled case
changes means only, never the actual sign-conditioned input law. No truth fit.

Write m=T-2>=2,n=m-1,F=Phi and Gplus=max(0,2F-1),Gminus=min(1,2F). Conditional
on sign, one omitted prefix has CDF G; the other n are iid normal. A=G F^n is
the probability all omitted inputs are below a threshold, and
B=(1-G)F^n+nG(1-F)F^(n-1) the probability exactly one is above. For suffix y,x,
first-suffix truth weight is A(y) above x>y and B(y) below; last-suffix truth
weight is A(x) below x<y and B(x) above. The special prefix label0 has weight
integral_L^U F(v)^n dG(v)+integral_U^infinity n(1-F(v))F(v)^(n-1)dG(v),
where L=min(x,y),U=max(x,y). Every other omitted label is exchangeable; its
weight can either be the remaining mass/n or, independently, the positive
integrals of G F^(n-1) over[L,U] and
(1-G)F^(n-1)+(n-1)G(1-F)F^(n-2) over[U,infinity]. For n1 omit the zero term
without evaluating a negative power. T55 independently checked the rank laws,
conditional mean factor and shuffled-control interpretation.

For numerical stability evaluate these positive polynomial integrals in
q=Phi(v), splitting at q=.5. For T4/T5 every integrand degree is at most3;
the two-point positive Gauss rule is an exact polynomial formula in real
arithmetic, not an unbounded quadrature approximation. Integrating the final
scalar x likewise splits at x=0 and x=y; on each piece all rank weights are
polynomials in Phi(x) of degree at most m<=3, so the same two-point formula
integrates them exactly. Use stable normal-CDF interval mass; count all
polynomial nodes/operations. Floating CDF/geometry error remains qualified.

Reuse the independently checked affine two-step arrangement and fixed
Gaussian-landmark panels read-only. Outer y[-6,6], production GL16, diagnostic
GL32, omitted accuracy at most2Phi(-6). No recursive hidden-state mesh or
new model-input sampling. Special-prefix rank labels must never be pooled
uniformly with the other omitted labels.

## Alternatives and falsifiers
1. A gate-history bit explains the lost information: true association should
   improve the prior failed case and outperform shuffled association.
2. Variation within each sign remains crucial: both mixtures can still fail,
   or shuffled can match/improve the true one. Keep that negative; do not fit
   mixture weights or another mean from accuracy.
3. Rank bookkeeping or integration is wrong: positive label weights must sum
   to1; averaging both sign laws must reproduce the original iid rank law;
   constants, zero histories and Wh0 must reproduce old results. Compare a
   scalar interval against independent direct-x numerical integration.
4. More components cost too much: count both arrangements/mean setup/rank
   integrations and report CPU/event work, not just doubled forward calls.

## First bounded cycle
Before trained scoring, verify rank-law identities on fixed grids/intervals,
including signs/tails, and the conditional constant-label0 law
P(correct|S=s)=2/T times the Beta(T-1,2) mass of the corresponding q half.
Their mixture is1/T; other fixed labels have(1-P0)/(T-1) for each sign.
Then use exactly the prior4 cases: T4base, T4relative .02copy0, T5base and
T5seed2, both true/shuffled variants at GL16/32. References only in scoring.
No expanded sweep until controls and the frozen rescue criterion pass.

## Status
Primitive and surrogate-driver controls pass; frozen trained pilot next.

## Results
Rank partition/averaging and independent nested direct-integral controls pass
(maximum independent difference 1.67e-16). Conditional constant-label, Wh0
and zero-history reproduction controls pass (maximum reproduction 5.56e-17).
T55 independently checked the positive regular-prefix formulas and exact
polynomial degrees. The initial proposed 80 scalar operations per rank query
was rejected by T55 as too small; it is removed, not used as a cost bound.
Initial passing artifacts with that invalid counter are retained under
initial_counter_*.json with the limitation stated. Corrected controls were
rerun; current results contain event counts only. Every numerical failure
and correction will be retained.

## Compute accounting
One thread,2GB cap,<=110CPU cycles. Review lock for<=20CPU controls; main lock
otherwise. Full per-estimate setup, inherited source hashes, all inner/outer
nodes and comparisons remain visible. Event counts are not a portable-FLOP
certificate. No persistent model-dependent cache.

## Claims
[inferred] These rank laws exactly score the two-sign surrogate, not the
original network. [speculative] The sign bit improves its accuracy/cost.

## Alternatives not ruled out
More conditioning, conditional variance, within-sign state support and the
rank-conditioned transport already studied separately. No general history
model is rejected if this finite mixture fails.

## Failure modes
Wrong special-prefix ownership, rank polynomial switches at0/y, CDF-tail
cancellation, affine floating geometry, and increased mixture cost.
