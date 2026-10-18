# T243 conditional residual tail prerequisite

## Question
What exact tail and second-moment information is available for the T233 residual,
and what additional information is needed to turn it into a decision bound?

## Method
`PROOF.md` conditions on the entire past, trims the threshold interval for an
exact Gaussian tail, differentiates a second-moment antiderivative, and gives a
deterministic winner-gap union bound. `manifest.json` pins reviewed predecessors.
Single replacement and shared subsequent inputs are explicit.

## Status
Bounded proof-only phase passed independent written review 50c6a8dd, published
shared c15f2eff. No numerical execution is authorized.

## Results
[inferred, independently reviewed] Exact conditional tail and cubic density bound;
small-band relative variance can diverge while absolute residual moments shrink.
A margin-gap plus tail bound needs no independence, but the gap distribution
remains unknown. No trained performance result is reported.

## Compute accounting
Written algebra and source/hash inventory only. Zero author/candidate/control,
Gaussian, target/checkpoint/array/model/geometry/mass/MC/RNG or fit calls.
No state distribution or margin probability is supplied for free.

## Claims
Exact real-arithmetic implications under the written hypotheses only. No stable
numerical evaluator, certified target accuracy, speed or full-cost inference.

## Alternatives not ruled out
Different anchors, multiple residuals with their cross-dependence, truncated
laws, sharper signed propagation, stable numerical tails and cheap gap bounds.
An uninformative bound is not a proof that those alternatives fail.

## Failure modes
Conditioning on ranks while retaining the untruncated normal law; endpoint/atom
confusion; cancellation in CDF differences; substituting covariance for a third
moment; outcome-dependent thresholds; and inferring accuracy from mean residual
without the missing margin information.
