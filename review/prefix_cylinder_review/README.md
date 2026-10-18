# Independent T213 prefix-cylinder theorem review

## Question
Does the published proof correctly show that exact vector-margin equality on a full-dimensional prefix cylinder is already global equality?

## Method
Written review of original cccd0f49, published20ed7393, delegatedd0e0cd94. Read its two pinned dependencies: T208 homogeneous hinge normalization/independence and T11597 fixed-weight identifiability. No author imports, symbolic engine, model/predicate, geometry or numerical calls. An independent vertical-line argument below checks the key mechanism.

## Status
Pass for the exact stated theorem and conditional fingerprint corollary. DRI confirmation remains separate. No stored-domain applicability or estimator result is asserted.

## Independent argument
Normalize every nonzero fresh coefficient to1, using ReLU(b*s)=|b|ReLU(s)+1_(b<0)b*s. Combine proportional hyperplanes, retain nonzero vector coefficients, and write the canonical remainder as A(z,t)+sum_j c_j ReLU(alpha_j·z+t), with distinct alpha_j. For two distinct alpha vectors, equality of their t-breakpoints at a given prefix means (alpha_i-alpha_j)·z=0, a proper prefix hyperplane. A finite union of these sets cannot cover the open ball in int(C). Choose one prefix z0 outside them. Along its full vertical line all hinge breakpoints are distinct. The derivative with respect to t jumps by exactly c_j at breakpoint j, while A has no jump. If the function is identically zero there, every c_j is zero. The remaining linear map vanishes on int(C)×R, an open set in the ambient space, so A=0 globally. This establishes the same conclusion without relying on the author's local ambient-ball/Jacobian construction.

The author's construction is also valid: nonzero fresh coefficient makes each hyperplane an invertible graph over prefix coordinates; its patch inside the cylinder is relatively open. Other distinct hyperplanes intersect it only in proper subspaces. A small ball avoids all other hinges and exhibits the nonzero outer-product jump. Neither convexity nor finite generation of C is needed. Boundary equality follows directly from global function identity and does not discard first-index output ties.

## Dependency and corollary check
Nonzero Wi makes every hinge in f_M-f_N eligible; canonical combination creates no new hyperplane with zero fresh coefficient. With separate T11597 assumptions, jump-coefficient subset injectivity identifies the same unit group at each hyperplane, then equality of normalized forms gives Wh*M=Wh*N and invertibility gives M=N. No new model predicates were computed. Failing any sufficient predicate remains inconclusive. Function/margin equality is stronger than label equality and the theorem does not establish a lower bound on accuracy-estimation complexity.

## Counterexamples checked
1. ReLU(-z) is zero for z>0 at every t but nonzero at z<0; its fresh coefficient is0.
2. ReLU(z+t) and ReLU(2z+t) vanish together when z>0,t<-2z, yet have values1 and2 at(1,0). The domain is full dimensional but not a cylinder.
3. ReLU(z+t)-ReLU(t) vanishes on z=0 for every t, but equals1 at(1,0); the prefix set has empty interior.
4. Scores(0,ReLU(t)) and(0,2ReLU(t)) predict0 for t<=0 and1 for t>0, including first-index ties, while their positive margins differ. Margin inequality need not mean label inequality.

## Applicability limits
An exact gates-only prefix history can define C×R because it precedes the last input. Initial full-sequence rank/truth charts may restrict that input; full dimension of a stored ray hull does not restore the cylinder premise. The proof explicitly preserves this distinction. Existing T11596 fixed represented-hull inequality rests on separate rank/sign/canonical evidence, not this corollary. Reset-identical maps, constrained fresh-input domains, lower-dimensional prefix sets, zero Wi, label-only or approximate equivalence remain outside the exclusion.

## Results and compute accounting
[inferred, independently checked] The written theorem and four counterexamples are sound under their stated real-arithmetic hypotheses. Only document/hash/Git provenance operations were performed; no candidate/reference/model/numerical/rank/mass/forward calls. Agent reasoning is not declared free production preprocessing. No performance, population mass, all-pair state census or challenge solution result follows.

## Alternatives not ruled out and failure modes
Applying the result to arbitrary full-dimensional stored cones, silently transferring trained fingerprint predicates, or replacing margin equality with label equality would be invalid. Future applicability needs its own exact domain/history and weight binding. No new empirical phase is authorized by this review.
