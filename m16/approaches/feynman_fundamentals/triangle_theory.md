# Two exact inner simplex coordinates

Let A,B,C in R^d be nonzero endpoints in a common closed orthant. Write
E=[A B C]. Initially assume rank(E)=3. Thin QR gives E=Q R with Q^T Q=I_3,
so ||E y||=||R y||. The barycentric domain is y_i>=0, sum(y_i)=1,
with ordinary coordinate measure dy_1 dy_2 of area1/2. Set c=R^{-T}1.
The affine plane containing z=R y satisfies c dot z=1.

Under radial projection omega=z/||z|| onto the unit sphere, the surface
Jacobian is

    dOmega = |det R|/||R y||^3 dy_1 dy_2.

One direct check uses z=R_3+y_1(R_1-R_3)+y_2(R_2-R_3): the scalar triple
product z dot [(R_1-R_3) cross (R_2-R_3)] is det R, independent of y.
The radial normalization supplies ||z||^-3. Since ||z||=1/(c dot omega),

    integral_simplex ||E y||^-d dy
      = |det R|^-1 integral_spherical_triangle (c dot omega)^(d-3) dOmega.

Rotate the unit vector c/||c|| to coordinate0. The remaining integral is
||c||^(d-3) times the degree-(d-3) moment already implemented in moments.py.
This also works with an integer exponent p>=3 different from the ambient
dimension, replacing d by p in the integrand and moment degree. There is no
extra factor2. For the last two Duffy coordinates, the Jacobian factor1-u
is precisely the coordinate measure of this triangle. Any factors depending
on earlier coordinates stay outside.

For orthogonal unit endpoints in any ambient dimension the exact integrals
for exponents3,4,5,6 are respectively

    pi/2, 3*pi/4, pi/2+2, 3*pi/2+3/4.

For example, these follow by expanding (omega_1+omega_2+omega_3)^(d-3)
over the positive spherical octant and using

    integral_octant x^a y^b z^c dOmega
      = Gamma((a+1)/2) Gamma((b+1)/2) Gamma((c+1)/2)
        / [4 Gamma((a+b+c+3)/2)].

The root reviewer and T48 independently checked the radial Jacobian and power.

## Stability and the deliberately inexact fallback

Thin QR avoids squaring the condition number as a Gram-matrix construction
would. Nevertheless tiny singular values make the plane normal and determinant
unreliable, and thin spherical triangles can make boundary moments cancel.
The prototype checks the ratio of smallest to largest singular values and
uses the analytic formula only above1e-5. It also checks positive upper/lower
bounds on both the affine integral and the spherical moment. These checks
detect some failures; they are NOT a floating-point error certificate.

Suspect and rank-deficient triangles are integrated in the ORIGINAL affine
coordinates, avoiding division by a tiny determinant. Let q(y)=||E y||^2,
choose valid positive lower/upper bounds qlo,qhi, mid=(qlo+qhi)/2, and
r=(qhi-qlo)/(qhi+qlo). For the integrand q^(-s), s=d/2, truncate its binomial
expansion around mid after degreeK. The uniform remainder is at most

    mid^-s * (s)_N r^N/N! / (1-rho),
    N=K+1, rho=r*(s+N)/(N+1)<1.

Under the Duffy map the truncated polynomial has degree at most2K+1 in each
coordinate including the Jacobian. An n-point tensor Gauss rule is exact for
it when K<=n-1. Positive weights sum to area1/2. True-integral and quadrature
remainders therefore add to at most the displayed uniform bound, with no
extra determinant. The implementation selects a rule up to128 nodes per
coordinate based on this bound, and raises if it cannot reach the tolerance.

The lower radial bound uses the minimum projection onto the normalized sum
of endpoints; convexity makes this valid. The upper bound is the largest
endpoint squared norm. The common-orthant condition ensures positive lower
projection. A global endpoint scale is factored out before either method.

Fallback output is NUMERICAL quadrature with an explicit quadrature-only bound.
It makes no claim of exact floating geometry/arithmetic, and the analytic path
also lacks a certified rounding bound. A caller must retain fallback counts,
nodes, and error contributions. No stochastic model evaluations are involved.

## Remaining work and cost

This operation reduces ordinary spherical-simplex outer integration from d-1
coordinates to d-3: one for T4, two for T5, three for T6. The positive exact
inner operator preserves degree in the remaining coordinates, so an existing
outer polynomial-approximation argument can survive this replacement.
Numerical fallback error must be added with the appropriate outer weights.

For the actual Duffy endpoints there is an additional stability simplification,
identified by the root reviewer. Original simplex rays have signed L1 norm1,
and every endpoint is a convex combination of those rays. The same orthant sign
vector s therefore satisfies s dot A=s dot B=s dot C=1. Supply s through
`known_plane_normal`; the primitive checks this residual and computes c=Q^T s
(with the internal global scale accounted for). This avoids solving R^T c=1.
In unscaled coordinates, 1<=||c||<=sqrt(d): projection cannot increase ||s||,
and c dot endpoint=1 with endpoint norm<=1 gives the lower bound. This controls
the plane-normal factor even for thin triangles. Determinant and spherical
moment cancellation still require separate treatment. Exactly rank-deficient
triangles still use the original-domain fallback.

The prototype is a scalar triangle API, not a production batched integrator.
Per analytic triangle it performs a thin QR, a3x3 singular-value decomposition,
a linear solve, rotation and normalization, one boundary moment recurrence,
and explicit geometric bound checks. Counts expose these operations and the
finite Fourier exponentials/sincs in the moment recurrence. Special-function
counts are at the Python-level API, excluding internal LAPACK implementations.
This overhead may outweigh the nodes saved on small cases; trained-model cost
comparison and vectorization remain essential before claiming an advantage.
