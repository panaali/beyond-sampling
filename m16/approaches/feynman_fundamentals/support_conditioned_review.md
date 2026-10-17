# T40 support-conditioned Gaussian endpoint review

Read-only review of `support_conditioned.py`, `exact_hidden_map.py`,
`transition_orthant.py` at f79a918, and its inherited bounded orthant
integrator. This is a finite Gaussian closure with support conditioning,
not a moment-consistent truncated-Gaussian fit or an exact neural law.

For full-column-rank Wo, applying its left inverse to saved conditional
logit mean/covariance recovers hidden mean/covariance in exact arithmetic.
The recorded condition number, reconstruction residual, symmetrization
and positive-definiteness checks are relevant numerical qualifications.
Mapping all rays of the convex gate/rank cone through its linear hidden
map gives the hidden cone; in two nonnegative dimensions its support is
the angular interval between extreme nonzero mapped rays.

The exact dyadic hidden-map computation and rank test are correct. A
nonzero rank-one map with nonnegative hidden output occupies a single
ray on its cell, so its readout winner is constant. Computing that winner
from an exact interior image preserves ties. This is conditional on the
floating cell construction identifying the intended gate region; checking
interior points is not a proof of the original numerical geometry.

For a rank-two Gaussian, the wedge normals
`(-sin(lo),cos(lo))` and `(sin(hi),-cos(hi))` correctly impose angular
support. Intersecting all readout margin halfspaces with that interval
gives the winner numerator. Strict versus weak boundaries have zero
Gaussian mass except identically zero margins, which the source handles
explicitly. Dividing numerator by support mass implements the stated
closure. The displayed ratio error expression
`(numerator_error+p*denominator_error)/(denominator-denominator_error)`
is valid when its component errors really are bounds. Adaptive error
estimates and rounded covariance/rays do not supply such certification.

`check_recent_aggregates.py` contracts the saved six rank-two rows and
constant correct mass independently in 60-digit Decimal arithmetic.
Orthant-only accuracy is 0.40919931833032920559 (bias -0.0009719433965);
component-cone accuracy is 0.41034753730671331252 (bias +0.0001762755799).
Seven additional residual components are exact rank-one winners. These
agree with the saved aggregates without importing the owner's code.
This is one-model finite improvement; full geometry/moment costs and
uncertified arithmetic remain, so it is not a general endpoint success.

The earlier bounded-bulk correction still missed a narrow conditional
CDF layer for a thin isotropic wedge: it returned zero with zero error
estimate instead of about 1.59e-5. The preserved transition-knot rescue
solves `(b+rho*z)/sqrt(1-rho^2)=t` for fixed t=-12,-8,-4,0,4,8,12 and
is algebraically correct. Near-unit-rho rounding remains distinct. The
width-1e-7 case takes a singular limit; its angular allowance is based
on the rounded correlation and is not a bound on covariance construction
error. Polar-angle controls owned by the author are a useful independent
integration-order falsifier. No claim that adaptive quadrature is now
universally reliable follows from these finite repairs.
