# Coupled finite-prefix ranks and adjacent-CDF failure

Read-only history_distribution/coupling.py review: for an existing target,
count the other already fixed inputs above it. Count0 requires final x above
target, count1 requires x below, and count>=2 cannot be second-largest.
For final target x, its valid interval is between the largest and second
largest of the other three values. The source implements these intervals.
Its >= convention for fresh y against an earlier-index fixed target agrees
with stable ascending input-index ties; fixed-prefix ties are rejected.
Continuous scalar x ties have zero Gaussian mass.

The y panels include prefix values, crossings of affine x bounds with either
prefix value, and crossings with x=y. Together with pre-existing network
pieces these cover the rank changes. Fixed-prefix label probabilities are
qa^2 for a lower prefix value, 2qa(1-qa) for the higher one; the remaining two
fresh labels share the remainder by exchangeability. These are valid controls
for a finite representative prefix, not a proof that the prefix rule has
the original Gaussian target frequencies.

The retained negative CDF interval is a numerical defect, not negative
probability or a failed coupling mechanism. Exact stored-float endpoints
lo=-0x1.e153eb75e622bp-1 and hi=-0x1.e153eb75e622ap-1 have true mass
2.84715341832977905212e-17. The old library CDF difference is-5.5511e-17.
Independent160-digit integration verifies the positive formula
delta*exp(-lo^2/2-lo*delta/2-delta^2/8)/sqrt(2*pi) to relative1.29e-16.
It avoids forming an unrepresentable midpoint. In real arithmetic the
midpoint quadrature remainder is bounded by delta^3 sup|phi''|/24, and the
endpoint-density lower/upper bounds also apply. Their exact width here is
2.97e-33, which ordinary floating endpoint evaluation need not resolve.
Thus the production fallback's floating bounds remain qualified rather than
being advertised as an outward enclosure. No negative mass is clipped or
discarded. This independent fixture check cost .03694 CPU seconds.
