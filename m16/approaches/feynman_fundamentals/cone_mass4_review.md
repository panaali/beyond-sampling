# T119 independent four-dimensional mass reference review

Read-only review of the pinned singular-endpoint mass implementation
and the different-reference code in `cone_mass4_review`. No shared
author source is edited, and the whole trained corpus is not rerun.

For covariance C(t)=I+t(C-I), Plackett differentiation gives
P=1/16+sum asin(rho_ij)/(8pi) plus
integral_0^1 sum rho_ij*asin(pc_kl.ij(t))/sqrt(1-t²*rho_ij²)/(4pi²).
The independently implemented conditional correlation uses the correct
Schur numerator and variances. For row rays R, normals are rows of
inverse(R^T), as used. Thus the50/70-digit reference follows a different
path from selecting one singular endpoint; it is not a high-precision
reimplementation of that endpoint's null-vector branch.

If all rays share one coordinate orthant, a single diagonal reflection
maps R to abs(R) and preserves isotropic Gaussian mass. The positive
L1-simplex reference is valid there. Its already-reviewed integration
bound still excludes determinant/input roundoff. Completed examples
include nearest420, whose author call raises but positive mass is
3.3284420355e-8; nearest10506, whose positive reference is2.66355056e-10
versus the author's-2.01679328e-7; and mixed-orthant nearest495, whose
independent50/70-digit references agree at8.4608273508e-10 versus a
negative author output. These establish specific numerical failures.

An asin input of-1 minus about2e-16 is a narrow floating-domain defect.
Clamping it can prevent that exception but does not validate the rest
of the mass or repair negative results. Persistence of a negative
result at n160 suggests that the tested quadrature refinement is not
the whole problem; it does not prove that every quadrature variant
fails. Near-root subtraction g±sqrt(g²+det*q), determinant/cofactor
conditioning and near-zero null-vector sign assignment remain possible
causes, not a confirmed unique diagnosis.

The signed facet decomposition is mathematically valid up to boundaries
when the chosen apex and original rays have a common positive
cross-section. The facet-normal apex supplies such a cross-section:
perturb it by a sufficiently small positive sum of facet normals to
obtain a linear functional positive on every ray and the apex. The
usual oriented simplex identity then applies. Numerically, however,
signed subtraction of larger piece masses can lose a tiny result;
being mathematically valid does not confer good relative conditioning.
Positive fallback, stable endpoint arithmetic and conditioned high-
precision evaluation remain open repairs coordinated with T109.

No broad negative about cone integration, complete cost claim, or
confirmation of the author's overall estimator follows from this review.

## Finite orthogonal rescue of the mixed-sign failure

The frozen rational Givens rotation c=255/257,s=32/257 has c^2+s^2=1.
An independent Fraction application to the stored dyadic rays of
nearest495 verifies every rotated coordinate has a common sign across
all rays. Isotropic Gaussian mass is invariant under this rotation,
and a global coordinate reflection then gives a positive-orthant cone.
This establishes the geometric premise exactly for the stored rays;
the numerical rotated arrays and quadrature remain floating arithmetic.

The positive result8.46082735079679e-10 agrees with the independently
reviewed MP result8.460827350796833e-10. The integration-only bound is
not a total floating certificate. Fixing the four exposed failures
leaves all corpus values nonnegative, but does not validate every
remaining source-positive value or certify the whole estimator cost.
No further theoretical concern with this finite geometry-only rescue
was found. The original negatives/domain exception remain evidence.

## T123 full frozen quadratic geometry inventory

The independent Fraction checker now verifies the stored dyadic rays for
all4033 native common-orthant cones and all1826 chosen rational rotations.
Every rotation has exact c^2+s^2=1 and every transformed cone shares one
coordinate orthant. The488 uncovered cones remain outside this positive
representation; the finite rotation list failing to cover them is not an
impossibility proof. Geometry checks and T122 raw checks together cost
.2774CPU. Numerical corpus results remain a separate pending review.
