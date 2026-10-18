# Exact network-output sufficiency fails for the advertised summary

This is a source/theorem implication of previously published, independently
verified T138 evidence, under scopef7509964. `t138_pins.json` and `prior_t138/`
freeze all premises. No new input, network evaluation, LP or numerical
certificate is generated. The trained example checkpoint is the prior fixed
SHA256498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb.

## Reused exact premises

Independent review8f90409 checked the exact binary-weight recurrence and
reverse-adjoint margin coefficients. At the stored rational crossing q:

* All160 gate preactivations are nonzero; the entire gate word is fixed in an
  open neighborhood of q.
* All inputs have strict rank order0,3,6,9,8,7,5,4,2,1. The three largest are
  q0=1, q3=63/64 and q6=31/32. The exact stored q9 is negative.
* Logits3 and6 are equal and exceed every other output by a strictly positive
  exact gap (display approximately2.43219660145).
* On this gate cell, margin logit3-minus-logit6 is affine-linear in input.
  Its coefficient for x1 is the positive stored rational
  26873360817781588461563771635068235129028167929568559025839863130356380632195968577521945199973/
  533996758980227520598755426542388028650676130589163192486760401955554931445160137505740521734144.

## Implication

Let S(x)=(p1,p2,case,m1,m2,x9), using the head0..8 definitions in frozen
state.py. Choose sufficiently small epsilon>0 and consider q+epsilon*e1 and
q-epsilon*e1. Such an epsilon exists because there are finitely many continuous
gate/rank/other-logit inequalities with strict slack at q. More explicitly,
within the prescribed gate cell each relevant inequality is affine; take
epsilon smaller than every positive slack divided by the absolute directional
coefficient when that coefficient is nonzero. Zero directional coefficients
place no restriction. Both sides then retain the same gate word, rank order
and dominance of logits3/6 over other outputs.

Input1 is not one of the head top two and is not input9. Hence S is identical
on both sides: p1=0,p2=3,case2,m1=1,m2=63/64 and the same exact q9. The positive
margin derivative makes network winner3 on the plus side and winner6 on the
minus side. Truth is3 on both sides; first-index tie ownership at q itself is
irrelevant to the two strict sides.

Therefore no function of S can equal this network's output or correctness at
every input. In particular an exact hidden reconstruction from S is impossible,
because its fixed readout would otherwise give the same output on these two
points. The three-feature lookup may still approximate the hidden state and
score98% on held-out targets; this proof says nothing about that numerical
claim, its error rate, or the quality of a fitted probabilistic summary.

## Scope

This proves pointwise non-sufficiency for the exact-real interpretation of the
fixed stored weights. Strictness also permits nearby points with the same
qualitative crossing, but no explicit new neighborhood radius, conditional
probability or useful error lower bound is computed here. The proof does not
claim the floating implementation changes labels at arbitrarily tiny epsilon.
It does not rule out retained weighted lower-input summaries: T138 itself
provided a useful local affine margin. No global compression or ARC estimator
family is rejected.
