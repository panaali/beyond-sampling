# Gaussian endpoint: support test, singular limit, and numerical falsifier

For a homogeneous cone with generating rays R, a readout margin nonnegative
on every ray is nonnegative everywhere. If it is positive on at least one
ray, its zero set is a proper face, hence null under the cone's continuous
input measure. Therefore gate_mixture.py's constant-winner criterion is
correct: all competitor margins nonnegative, with a positive ray against
each earlier-index competitor. Identically zero margins against later
indices are allowed. This is a real-arithmetic support argument, not a
certificate for signs of floating dot products. Weighted fixed winners use
their full joint component mass, while residual endpoint laws are normalized
within each component. No extra target-prior factor belongs in that sum.

For two Gaussian differences, integrating phi(z)*Phi((b+rho*z)/sqrt(1-rho^2))
from z=-a to infinity gives the orthant probability. At rho=1 the limit is
Phi(min(a,b)); at rho=-1 it is the interval mass Phi(b)-Phi(-a), clipped by
empty-interval logic. Plackett's derivative is at most
1/(2*pi*sqrt(1-rho^2)), so replacing an admissible rho by its signed unit
limit costs at most acos(abs(rho))/(2*pi), for any means. An excursion outside
[-1,1] is not covered by that bound. A zero difference variance also requires
zero crosscovariance for a valid Gaussian; malformed matrices must not be
silently treated as deterministic differences.

Independent check_gaussian_tail_review.py exposed an actual integration
failure in the original full_covariance.py. With independent differences
mu=(40,0), covarianceI, exact probability isPhi(40)/2=.5, but the routine
returns2.75e-28 with reported quadrature error0. At mu=(100,0) and(1000,0)
it returns0. The broad semi-infinite interval lets adaptive quadrature miss
the Gaussian bulk. Tests at means0 and8 pass, so ordinary controls alone
did not falsify this defect. The fixed five-case check took .1100 CPU seconds.

A separate bounded-domain rescue is mathematically valid: intersect the
integration domain with[-12,12], split at physical bulk points and the
conditional-CDF transition, and bound omitted exterior mass by2Phi(-12)
because the integrand lies between0 andphi. An empty bounded interval does
not erase that omission bound. Numerical interior quadrature error is still
not an interval certificate. The owner preserved original artifacts and
reports corrected full_covariance_bounded.py reproduces all27 trained/control
laws, changing pooled accuracy only1.05e-15, with extreme controls passing.
This is a repaired implementation defect, not a Gaussian-law impossibility
or grounds for dismissing the gate-conditioned rescue.
