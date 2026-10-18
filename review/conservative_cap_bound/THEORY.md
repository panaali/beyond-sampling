# T200 represented-cone upper bound

Inputs are finite binary64 ray rows in R^d, d in{2,3,4,5}. No unit-ray
assumption is needed. Interpret every stored float as its exact dyadic rational.
Choose exact a=sum r_i. If a=0, any ray=0 or any a.r_i<=0, return1. This is a
safe uninformative bound; no claim the whole cone has mass1. Unsupported shape,
dimension, nonfinite input or resource limits also return1 with a reason, and
do not establish a finite-input mathematical assertion for an undefined object.

Otherwise A=a.a>0, R_i=r_i.r_i>0 and
q=min_i (a.r_i)^2/(A R_i) lies in(0,1] by exact Cauchy–Schwarz. For x=sum λ_i r_i,
λ_i>=0, a.x >=sqrt(q)||a||sum λ_i||r_i||>=sqrt(q)||a||||x||. Every nonzero x in
the represented positive hull lies in the cap with cosine c=sqrt(q)>0.
Let δ=1-q. This is a cone-hull theorem, not proof of algorithmic domain coverage,
disjointness, neural correctness or other estimator errors.

For isotropic Gaussian direction, spherical cap probability is at most:
- d2: sqrt(δ)/2, since asin(z)<=πz/2 on[0,1].
- d3: δ/2, since exact cap=(1-c)/2 and1-c<=δ.
- d4: 2δsqrt(δ)/3, since marginal density is(2/π)sqrt(1-z²),
  2/π<2/3 and interval[c,1] has length1-c<=δ.
- d5: 3δ²/4, since marginal density is3(1-z²)/4 and1-c<=δ.
The cap lies in a hemisphere, so taking min(the displayed bound,1/2) is safe.
These intentionally loose elementary bounds avoid incomplete beta evaluation.
At q=1 they correctly give0: all positive-dot rays are collinear, hence their
positive hull has Gaussian measure0 in every supported dimension.

A rational upper sqrt uses k=40: n=ceil(δ*2^(2k)), m=ceil(sqrt(n)) from integer
isqrt and an exact square comparison, U=m/2^k. Thus U>=sqrt(δ); substitute U
only into nonnegative expressions above. Arithmetic is exact rational/integer;
fixed bit/ray/operation caps may only replace a result with the safe fallback1.
No speed, as-implemented FLOP or scalable-region claim follows from exactness.

For optional binary64 output, convert the rational bound, compare the exact
stored dyadic float against the rational and move one nextafter step toward+∞
if needed. Check the resulting dyadic is>=bound and<=1. A failed conversion
check returns1. This also handles positive underflow tozero without silently
reportingzero. The rational comparison is the certificate; a library claim of
correct rounding alone is not relied on for acceptance.

Alternatives/confounds: (1) exact rational preprocessing may cost more than
saved work; first controls record CPU and expression/bit inventories, no win
inference. (2) elementary bound may be much looser than true cap; compare all
fixed orthants and original narrow fixture, retaining looseness. (3) rounding
repair can succeed while represented-domain or other estimator errors remain;
those are explicitly outside this primitive. No trained experiment in phase1.
