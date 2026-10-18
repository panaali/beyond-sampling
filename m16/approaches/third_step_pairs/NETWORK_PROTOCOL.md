# Analytic network-pair phase

Claims7d394ba1/2678e75d precede implementation. The pointed cone primitive and
24 controls are unchanged and independently reviewed84e0040c. Reuse only the
six exact H4 sector/map and mean certificates independently7ad8ea7a; no trained
weight/checkpoint/reference loading and no new forward-map work is needed.
For each of340 sector/upper-triangle unit pairs, a zero coefficient vector
implies an exact zero product before checking cone rank. Other products use
the sector constraints plus both output-positive constraints. Rank<3 remains
explicit unsupported; any such control outcome stops and is preserved.

Candidate cone moments and contraction use80/120 digits. Upper-triangle
network matrices are mirrored by construction; this does not independently
prove symmetry. Primitive symmetry already had its own controls. Sum sector
pairs, subtract the independently checked means, and inspect finite4D
principal-minor tolerances. No outward covariance certificate is inferred.

The independent reference uses G=(Rcos(theta),Rsin(theta),tR). Gaussian
Jacobian R² and pair product R² yield3/(4pi)*(1+t²)^(-5/2) after radial
integration. With A=a cos(theta)+b sin(theta), C similarly, the joint-positive
t interval integrates AC I0+(Ad+Cb)I1+bd I2, where I0=s-s³/3,
I1=-1/[3(1+t²)^(3/2)], I2=s³/3 and s=t/sqrt(1+t²). Infinity limits are exact.
All zero and threshold-intersection harmonic coefficients are exact Fractions;
root membership and duplicate ordering use exact halfplane cross signs before
MP angles. Constants, empty/tied/unbounded intervals and opposite/zero products
remain explicit; signed primitive cancellation is not clipped.

First eight reference-only controls all use exactly X=x1>=0, x2 free and
Z=x3 free (theta from-pi/2 to pi/2): ±Z positive squares1/4, opposite0,
X²1[X>=0]=1/2, X+Z+ cross1/(2pi), negative-X timesZ+ zero,
(X+Z)+(X-Z)+=1/pi, zero product0. This is one halfplane, not an extra
independent sign restriction. Fixed reference precision80, analytic160,
absolute reference-domain tolerance1e-50. No adaptation from observed failure.

Then all340 analytic network cells compare with reference at absolute1e-50
plus relative1e-40, candidate80/120 stability1e-60*(1+abs(value)). Known raw
matrices for zero and zero recurrent networks are checked. Identity recurrent
diagonals use Wi_i²*[3/2+(1+sqrt2)/(2pi)] from the finite Spitzer second
moment formula; cross pairs still require independent angular references.
Reference/geometry/candidate exact arithmetic and MP/validation costs remain
separate. One primary110CPU attempt, one thread2GB, durable per-reference and
per-cell attempts/results, refuse existing files. No trained covariance,
later propagation, final accuracy, new random input or fullcost/FP claim.
