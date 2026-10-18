# Independent T243 residual-tail written review

## Question
Do the conditional tail, second moment and downstream disagreement statements
follow from the reviewed single-coordinate two-hinge residual identity?

## Method
Written review of author1106dbb4 and clarification77e5f473 under delegated claim ac8bf56b. The metadata
checker binds proof and predecessor text only; it imports no author code and
evaluates no Gaussian function, model or scientific fixture.

## Status
Pass as an exact-real conditional theorem. No numerical execution authorized.

## Results
On the mismatch interval the residual is B times distance from the endpoint v.
For 0<=eta<B*delta, its strict superlevel set is the open interval from
v+sign(u-v)*eta/B to u. The absolute CDF difference is therefore correct in
either orientation. At eta=0 the atom at residual zero is excluded; at the
upper support endpoint the strict tail vanishes. The a=0 pointwise ownership
remains in the inherited exact predicate even though its conditional Gaussian
probability is zero. Coincident roots give residual identically zero.

Differentiating F_v cancels its constant and linear density terms, leaving
(x-v)^2*phi(x). Integrating over a length-delta interval from endpoint v gives
the second-moment identity and cubic bound B^2*delta^3/(3*sqrt(2*pi)). The
alternative B^2*delta^2 times support probability and Markov bound are valid.
Taking expectation requires an absolute directional third moment, which a
covariance alone does not determine. The separately stated weaker bound using
B^2 E[delta^2] is valid from support probability at most one; it needs raw
second moments (mean as well as covariance). The support-tail width times
maximum density gives (delta-eta/B)/sqrt(2*pi) within the support. No stable
floating subtraction follows.

For fixed finite v and fixed B, substitute x=v+sign(u-v)*delta*t, 0<t<1.
The density converges uniformly to phi(v) on this shrinking interval. Integrals
of 1,t,t^2 give the three asymptotic constants. Dividing the second moment by
the squared mean yields 4/(3*phi(v)*delta); subtracting one gives the same
leading relative-variance asymptotic. It is not uniform when v or B varies.

Coordinatewise ReLU Lipschitzness propagates a single initial residual through
identical future inputs, even from a signed surrogate state. Taking the maximum
over all distinct output pairs makes K deterministic and valid for the random
surrogate winner. If G>K*eta and r<=eta, every original winner-versus-rival
margin remains strictly positive. Thus the stated event inclusion and union
bound require no independence between G,r or the task label. Past-measurable
thresholds permit conditioning; thresholds selected from the current input or
future logits do not automatically retain the fixed-threshold CDF expression.
For K=0 all pairwise margins, including ties, are identical and disagreement is
empty. The unknown small-gap term remains an essential unsolved dependency.

Positive diagonal change of coordinates scales r and eta by D_j and every
pairwise K by 1/D_j. Consequently the tail event, weighted second moment,
surrogate logits and gap are invariant. This reparameterizes weights; it does
not contradict the separate fixed-weight symmetry result.

## Compute accounting
Text/Git/hash operations and symbolic reasoning only. No author imports,
Gaussian numerical calls, target arrays, model, RNG, fit, mass or geometry.

## Claims
[inferred] The stated conditional identities and inclusions pass under finite
opposite nonzero slopes, fresh independent normal input, one replacement and
common future input assumptions. [measured] Pinned text bytes and prior review
publication are checked. No target probability, cheap gap law, cost advantage
or estimator claim follows.

## Alternatives not ruled out
Truncated input laws, zero/same slopes, multi-coordinate schemes, stable
evaluation and stronger certified margin distributions remain separate tasks.

## Failure modes
Ignoring the zero atom, swapping strict tails at zero, using covariance as a
third-moment bound, selecting thresholds from the scored input without new
analysis, or using a winner-specific random coefficient as a deterministic K.
