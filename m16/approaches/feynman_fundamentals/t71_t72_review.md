# Independent review of odd-dimensional reduction and enclosed T4 mass

T55 review, 2026-10-17. T71 and T72 retain implementation ownership. This review
does not change their engines or certify the original H16,T10 challenge.

## T71: measure, signs and conditioning

For four endpoint columns E in R5, write E=QR with Q orthonormal, R square,
and c=R^(-T)1. Barycentric coordinates y live in the ordinary parameter
tetrahedron of volume 1/6. The radial map has

    dOmega_3 = |det R| / ||Ry||^4 dy.

Consequently the inverse-fifth integral is

    integral_Delta3 ||Ey||^-5 dy
      = ||c|| / |det R| integral_spherical_tetra u_0 dOmega_3,

where u_0 is measured along c. On S3, Delta u_0=-3u_0. The divergence theorem
therefore gives M1=-(1/3) sum_i nu_out,i,0 * area(face_i). There is no extra
factorial. If w_i is the cofactor vector satisfying w_i dot R_j=det(R) delta_ij,
then the inward face normal is sign(det R)w_i/||w_i||. Thus the implementation's
sum sign(det R)(w_i dot c)/||w_i|| * area(face_i)/(3|det R|) has the right sign
and normalization. The cofactor indexing in the reviewed primitive agrees.

For each face with three unit rays a,b,d, its S2 area is

    2 atan2(||a wedge b wedge d||, 1+a.b+b.d+d.a).

The cofactor norm divided by the three endpoint norms is precisely the wedge
norm. Common-orthant endpoints make the denominator at least one. The numerator
can nevertheless be inaccurate for thin faces; a negative rounded Gram
determinant cannot simply be clipped to zero as evidence of zero mass.

With a supplied affine normal s satisfying s dot E_i=1, c=Q^T s avoids an
ill-conditioned solve. For signed-L1 endpoints 1<=||c||<=sqrt(5). Under the
implementation's coordinate division by scale a, the projected normal must be
Q^T(a*s), exactly as implemented. Dividing the final value by a^5 restores
units. A residual threshold checks a numerical precondition, not exact validity
of a supplied plane normal.

The current singular-ratio and cancellation screens are useful rejection rules,
not rounding certificates. In particular, cancellation among facet terms can
hide behind a nearly singular coordinate calculation before the final sum.
Explicitly keep this qualification even when independent controls pass.

## T71: positive fallback and falsifiers

The fallback's two Jacobi variables carry weights (1-u)^2 and (1-v); the last
segment is integrated by I5. A degree-K polynomial in the squared norm has
degree at most 2K in each Duffy coordinate. Jacobi order n therefore permits
K=n-1. The exact positive segment integral preserves this polynomial argument.
If E_K bounds the uniform function remainder, the true and quadrature errors
sum to at most 2*volume(Delta3)*E_K=E_K/3. This matches the reviewed fallback.
Its ordinary floating computation and I5 arithmetic are not enclosed by that
real-arithmetic truncation bound.

Three decisive controls, with different failure mechanisms:

* Four orthogonal unit endpoints: integral=2*pi/3. Each positive-orthant S3
  coordinate integral is pi/6, and the plane numerator sums four coordinates.
* Equal endpoints p: integral=||p||^-5/6, including rank-deficient endpoint
  matrices. With p=(1/4,1/4,1/4,1/4,0), this is 16/3. Zero inner determinant
  does not justify discarding this parameter integral. A zero determinant of
  the original full-dimensional probability cone is a separate issue.
* Asymmetric endpoint coalescence with the perpendicular projection of the
  origin outside the tetrahedron: individual face contributions are order
  epsilon^2 but their sum can be order epsilon^3. A symmetric centroid example
  misses this cancellation. Compare permutations and positive rescalings with
  original-domain positive cubature, and preserve every rejected/failed case.

Higher odd spherical moments obey

    M_p = ((p-1)M_(p-2) - sum_faces nu_0 F_(p-1))/(p+2).

Odd degrees terminate at M1 and use S2 face moments. Even degrees require the
S3 volume M0, so the T5 shortcut does not automatically remove three numerical
coordinates for T6.

## T72: enclosure assumptions and analytic remainder

Reviewed source: enclosed_mass.py, with the exact SHA recorded by
check_enclosed_review.py. Integer ray rows are interpreted as exact integers;
L1 normalization and the absolute determinant must be exact rationals from
the same simplex. Source topology correctness remains a separate dependency.

The rational Legendre construction provides n disjoint sign-changing intervals
inside (-1,1). A polynomial of degree n has at most n roots, so those intervals
account for every root. Rational interval recurrence encloses P_(n-1), and
the root identity gives weights 2(1-x^2)/(n^2 P_(n-1)(x)^2). A denominator
interval containing zero must fail rather than manufacture a weight. Exact
rational transformations of nodes and weights to [0,1] preserve containment.

The alternating Machin formula encloses pi rationally. Conversion of each
rational endpoint to binary64 is checked against Fraction.from_float, not
assumed to round in the desired direction. Positive ray coordinates allow
monotone interval products, squares and division. Every reduction level uses
outward elementary operations; a single nextafter after an opaque dot product
would not have been enough. Intersecting a lower bound with zero is legitimate
only because each represented operation is known nonnegative.

Assumptions: binary64 elementary add/subtract/multiply/divide implement the
usual correctly rounded IEEE behavior, nextafter reaches the adjacent value,
and no unsafe compiler transformations or flush-to-zero mode invalidates those
properties. Finiteness must be checked on outward endpoints as well as the
pre-rounded operation. The fixed T4 domain is far from overflow/underflow, but
generic use should reject a nextafter-generated infinite endpoint.

For signed-L1 rays in R4, q=||Vy||^2 belongs to [1/4,1]. The exact Chebyshev
coefficients of q^-2 have magnitudes

    |c_k| = 8 * 3^(-k) * (k+5/4), k>=1.

Tensor Legendre order n absorbs the Duffy Jacobian as an ordinary polynomial;
its largest coordinate degree is 2K+2. Therefore K=n-2 is valid. The positive
tail starting at N=n-1 sums exactly to

    E_K = (12*n+9) / 3^(n-1).

Both true and positive cubature integrate the tetrahedron's Jacobian to 1/6,
so their error is at most E_K/3. Multiplying by |det V|/(2*pi^2) gives precisely
the implemented |det V|*E_K/(6*pi^2). The remainder itself is rational before
the already-enclosed pi factor: there is no unguarded floating special-function
bound hidden in this last step.

Given a raw quadrature interval [L,U] and outward remainder e, [L-e,U+e]
encloses the exact cone mass under these assumptions. Summation of cone values
and errors also must be directed. The exact orthant probability 1/16 need not
belong to the raw finite-rule interval; it must belong to the interval enlarged
by the analytic remainder. Confusing these is a particularly useful falsifier.

## Independent checks and limits

check_enclosed_review.py independently generates a 70-digit GL20 rule and
evaluates its tensor cubature on the orthant and fixed exact-artifact cones 0
and 128. It computes integer determinants by a separate permutation expansion
and compares with the stored rational determinant. It tests raw quadrature
containment, not agreement with the exact probability; all costs and hashes
are recorded in enclosed_review_results.json.

High-precision agreement is a falsifier for implementation errors, not a proof
that mpmath itself supplies directed enclosures. The mathematical operation
audit plus exact setup supplies the enclosure argument. The checks do not
replace independent topology review or certify the complete model family.

Measured results: all three GL20 raw intervals contained the independent
70-digit values (24000 nodes, 1.530285 CPU seconds). The orthant GL20 value was
0.0625000000000014896492789, illustrating the distinction above.

The asymmetric-coalescence test also ran independently at epsilon=1e-2,...,1e-8.
Two analytic cases agreed within8.947e-13 relative; their largest permutation
difference was2.324e-12. Five narrower cases were rejected by the cancellation
or singular-ratio guard and used positive fallback, agreeing within3.416e-16.
Cost12096 high-precision GL12 nodes and1.095377 CPU seconds. Source hashes and
all branches/errors are preserved in tetra_adversarial_results.json. Tiny
fallback quadrature bounds do not cover floating discrepancies, consistent
with the stated real-arithmetic scope. This test found no change needed to
the current heuristic policy, but cannot certify it for every endpoint array.

## T71 fixed-tangent follow-up

For one cone, choose an orthonormal frame of the three differences among the
last four rays. At outer node u, put L=1-u and write endpoint coordinates as
(delta,t_i(u)), where t_i(u)=u*t_0+L*t_i. The perpendicular component has length
delta>=1/sqrt(5) for signed-L1 rays. Its unit direction is the first coordinate;
the affine-plane normal in these coordinates is c=(1/delta,0,0,0).

Let D=det[t_2-t_1,t_3-t_1,t_4-t_1]. For omitted face i, take its remaining
indices j<k<l in their original order, n_i=(t_k-t_j) cross (t_l-t_j), and
h_i=t_j(u) dot n_i. Its omitted-column cofactor is

    w_i=(-1)^i L^2 (h_i,-delta*n_i).

Also det R=delta*L^3*D. Substituting these two identities into the independently
checked first-moment formula yields the implemented denominator
3*delta^2*L^3*|D| and normalized face coefficient
sign(D)*(-1)^i*h_i/sqrt(h_i^2+delta^2*||n_i||^2). Face areas use L^2 times
that square root in the atan2 numerator. Thus no QR or SVD is needed at each
outer node; only the per-cone tangent frame is reused. At L=0 the positive
equal-endpoint parameter integral must be retained, not evaluated by division
by L^3. The present API excludes endpoints u=0,1.

Subtracting the first column from the other three gives B=R*C, where C has
Frobenius norm sqrt(7). Since R^-1=C*B^-1,

    sigma_min(R)/sigma_max(R)
       >= 1/(sqrt(7)*||B^-1||_F*||R||_F).

The block inverse of B supplies the inexpensive formula in fixed_tangent.py.
This is a conservative real-arithmetic lower bound; a floating implementation
does not automatically certify it. A small computed lower bound may reject a
well-conditioned case, which costs time but preserves the intended fallback.

Independent70-digit original-domain GL12 checks at u=.99 and.999 on the
asymmetric cone found relative errors4.360e-14 and2.650e-14, with permutation
differences at most3.931e-14. Both used the analytic branch. Cost3456 reference
nodes and.410647 CPU seconds, recorded with exact source hashes in
fixed_tangent_review_results.json. No production code was changed.
