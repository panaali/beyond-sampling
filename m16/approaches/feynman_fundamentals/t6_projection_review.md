# T52 T6 projection and triangle accounting review

Independent read-only T55 review, 2026-10-17. No new integration was performed.
The individual-simplex perturbation algebra is correct. The static10000-unit
triangle allowance is conservative under the declared scalar/TRANS20 model;
the separately charged fallback has no hidden rule-generation kernel. No issue
was found requiring the ongoing bounded full-base calculation to stop.

## Projection perturbation

Let R,S be original/projected6-by6 ray matrices, with column differences bounded
by epsilon_j and epsilon=max_j epsilon_j. Let M bound every original/projected
column's Euclidean norm. Telescoping the determinant one column at a time and
using Hadamard gives

    ||det R|-|det S|| <= |det R-det S| <= M^5 sum_j epsilon_j.

For barycentric y in Delta5, the projected point S*y belongs to one orthant,
so its L1 norm is the barycentric average of the column L1 norms. Thus

    ||S*y|| >= min_j ||s_j||_1 / sqrt(6).

Furthermore ||(R-S)y||<=epsilon. Every point on the segment between R*y and S*y
therefore has norm at least

    a=min_j ||s_j||_1/sqrt(6)-epsilon.

If a>0, the gradient norm of f(z)=||z||^-6 is6||z||^-7, giving
|f(R*y)-f(S*y)|<=6*epsilon*a^-7 and f(R*y)<=a^-6. The absolute determinant of
S is at most M^6. For a6-dimensional Gaussian cone the mass is

    |det R|/pi^3 * integral_Delta5 ||R*y||^-6 dy.

Since volume(Delta5)=1/120, the mass difference is at most

    [M^5*(sum epsilon_j)*a^-6 + M^6*6*epsilon*a^-7]/(120*pi^3).

This is the implemented expression. It bounds absolute determinant differences,
not signed oriented volumes. It remains valid if a projected determinant
collapses to zero. Projected cones need not remain disjoint: sum these absolute
individual errors to compare the original sum with the projected sum. That is
not a new proof that the original floating triangulation represents the desired
region correctly. The original cones and every projected collapse must remain
in the provenance, as the owner has specified.

The2048-per-cone projection allowance covers fixed6-by6 normalization, errors,
extrema,18 column square roots, four norm powers and diagnostics with substantial
slack. Evaluated error bounds use ordinary floating arithmetic; the existing
roundoff_certified=False qualification is necessary. This review checks the
real-arithmetic inequality, not outward evaluation of its endpoints.

## Pinned triangle source and static allowance

The reviewed batch source has exponent6, hence moment degree3. All work before
the fallback is fixed-width array arithmetic. It constructs a two-vector frame
in R6, three local3D rays, three spherical edges and a degree3 moment. It uses
no QR, SVD, linear solve, eigensolver or iterative special function.

A deliberately loose independent partition of its work is below. Counts include
mask tests and reductions; square roots/trigonometric functions/general powers
are listed separately and cost20 each under the declared convention.

| Group | Ordinary-operation allowance per attempted triangle | Special-function allowance |
| --- | ---: | ---: |
| Input checks, frame and ambient norms |600|6|
| Local rays, edge geometry, area and first moment |400|10|
| Degree3 moment including all small-angle polynomial branches |600|40|
| Range/cancellation screens and metadata reductions |400|10|

Even these loose allowances total2000+66*20=3320, well below10000. All attempted
triangles receive the full charge, including those rejected before or after
moment evaluation. The additional10000 per batch covers scalar setup and empty
array/control handling. Per-fallback work is charged in addition, rather than
hidden inside this allowance. Source parsing, memory copies and interpreter
dispatch follow the same exclusions as the existing T52 operation model.

## Local positive fallback

Projected endpoints genuinely share an orthant. Their centroid dot products
are nonnegative, so squaring their minimum is valid in this restricted callback.
The signed-plane lower bound is combined with it. The midpoint expansion of
q^-3 has first omitted coefficient (3)_n/n!=(n+1)(n+2)/2. For a degree n-1
polynomial in q, Duffy degree2n-1 including its Jacobian is integrated exactly
by order-n GL. Subsequent coefficient ratios are bounded by
r*(n+3)/(n+1). Hence the implemented positive remainder formula is correct.

The fixed radial setup512, trial120 and node100 allowances are conservative.
For each quadrature node, the6-by3 matrix-vector product costs30, its norm
squared11, reciprocal cube3, and weighted summation about2; these are below100.
The scalar Newton Gauss rule and Duffy table setup are separately charged and
cached only inside one Engine. Every resumed Engine therefore pays cold setup.
There is no gamma or opaque eigensolver in this callback.

The output's analytic small-angle series bound, positive quadrature bound and
projection perturbation bound remain distinct. None encloses the heuristic
floating moment screen or the original DD topology automatically.

## Scope and source identity

Reviewed files are portable_cost/t6/projection.py and triangle.py in the
recover-small_h2_34 worktree. The adapter checks the exact T55 source digest
ed72ccfb52ee6f43afc19384dcc7f164f5022dc7c2610ba94da52f58b8e81461 before loading
triangle_batch.py, and replaces only its local positive fallback import. These
findings do not automatically cover a future changed source. Checkpoint/source
hash validation and completed-index continuity remain the full-run driver's
responsibility.
