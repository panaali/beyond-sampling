# A linear image of a normalized Gaussian: surface versus volume

Let X~N(0,I_k), R=||X||, U=X/R, and Z=B U with B an m-by-k matrix of
rank r. U is uniform on S^(k-1), independent of R. A thin SVD is
B=P Sigma V^T, with positive singular values sigma_1,...,sigma_r. In the
r-dimensional image subspace, use coordinates v=Sigma^-1 P^T z.

## k>r: an ellipsoid ball density

If k>r>=1, Z has density with respect to r-dimensional Euclidean volume on
the image subspace:

    f(z) = Gamma(k/2) / [pi^(r/2) Gamma((k-r)/2) prod(sigma_i)]
           * (1-||v||^2)^((k-r-2)/2),       ||v||<1.

This follows from the first r coordinates of a uniform sphere direction.
Their squared radius is Beta(r/2,(k-r)/2), and their direction is uniform on
S^(r-1). Equivalently, split an isotropic Gaussian into its r-dimensional image
coordinates and k-r-dimensional kernel coordinates, whose squared norms are
independent chi-squares. The linear coordinate volume factor is prod(sigma_i).

At k=r+1 the density diverges at the boundary like inverse square root;
at k=r+2 it is uniform on the ellipsoid ball; at k>r+2 it vanishes at the
boundary. Thus merely obtaining full-dimensional volume does not establish
a smooth bounded density suitable for a coarse Cartesian grid.

Examples with unit singular values:

- r1,k2: arcsine density 1/[pi*sqrt(1-v^2)] on(-1,1).
- r2,k3: disk density 1/[2*pi*sqrt(1-||v||^2)].
- r2,k4: uniform disk density1/pi.
- r3,k4: ball density1/[pi^2*sqrt(1-||v||^2)].
- r3,k5: uniform ball density3/(4*pi).

## k=r: an ellipsoid surface, not a volume density

When k=r, there are no kernel coordinates and ||v||=1 identically. The support
is an ellipsoid SURFACE of dimension r-1, even if B has rank r=m. Its density
with respect to induced (r-1)-dimensional surface area is

    f_surface(z) = 1 / [area(S^(r-1)) prod(sigma_i) ||Sigma^-1 v||].

The surface Jacobian of v -> Sigma v is
prod(sigma_i)*||Sigma^-T v||. This can be derived by extending an orthonormal
tangent basis at v with its normal v, or by the cofactor formula for an area
element. P is an isometry and contributes no further factor. Here
area(S^(r-1))=2*pi^(r/2)/Gamma(r/2). For r=k=1 the sphere S0 consists of two
points and the formula reduces to probability1/2 on each point, interpreting
surface area as counting measure. If r=0, Z is an atom at zero instead.

The common mistake is to infer volume dimension from rank(B) alone. The domain
is a sphere of dimension k-1, so no linear map can create a k-dimensional
volume measure from it. If a gate branch has r=k=3, its normalized image is
a two-dimensional ellipsoid surface in three-dimensional state space.

## Gate restrictions introduce a conditional fiber factor

On a ReLU gate cell, B is fixed but the input direction is restricted to a
conic region C intersected with the sphere. Therefore the unrestricted density
above cannot be used by itself. For k>r, extend V to an orthogonal basis with
kernel columns V_perp. Conditional on projected coordinates v, the original
direction has the form

    U = V v + V_perp sqrt(1-||v||^2) eta,
    eta uniform on S^(k-r-1).

The gate-cell SUBDENSITY is f(z) times

    g_C(v) = P_eta[ V v + V_perp sqrt(1-||v||^2) eta belongs to C ].

Do not divide by P(C) unless a normalized conditional branch law is intended.
For k=r, there is no fiber: multiply the surface density by1[V v belongs to C].
Distinct gate cells can overlap in the image, and their subdensities must be
added. Count/order constraints also enter C and cannot be discarded.

For k-r=1 the fiber is the two-point sphere S0: g_C is the average of two
indicators, taking values0,1/2,1. For k-r=2 the fiber is a circle and g_C is
an accepted arc-length fraction. This low-dimensional fiber structure is
potentially useful, but gate constraints remain essential.

An explicit counterexample: k2,r1, B=(1,0), and gate C={U2>U1}. Let v=U1.
The unrestricted density is arcsine. The conditional fiber acceptance is

    g(v)=1 for v<-1/sqrt(2),
         1/2 for -1/sqrt(2)<v<1/sqrt(2),
         0 for v>1/sqrt(2).

The branch has mass1/2 and first submoment -1/(pi*sqrt(2)). Treating the gate
as an independent half-probability would get the total mass right but wrongly
give first moment zero. This is precisely the kind of residual-independence
shortcut that must not re-enter a forward transport construction.

## Consequences for the H2 augmented normalized state

With target coordinate included, z=(h1/R,h2/R,u/R) is a gate-wise linear image
of the k Gaussian coordinates accumulated in R. Consequently:

- k1: at most finitely many points (initially target sign +/-1).
- k2 and branch rank2: ellipse arcs of dimension1, not planar area density.
- k3 and branch rank3: ellipsoid patches of dimension2, not3D volume.
- k4 and branch rank3:3D volume is possible, but its uncut base density has
  an inverse-square-root boundary singularity and a two-point gate fiber.
- k5 and branch rank3: the uncut base density is uniform, but gate-cell
  fiber fractions can still vary and add nonsmooth boundaries.

Branches of smaller rank retain their lower-dimensional ball/surface measures
at the same stages. ReLU collapse can create such branches repeatedly, so early
low-dimensional supports do not necessarily disappear when k grows.

The previous unnormalized H2 support theorem concerned h, whose radius has
already been integrated and produces ray/area densities. It must not be
transferred unchanged to h/R or (h/R,u/R). For h/R alone, rank2 at k2 already
means a one-dimensional ellipse boundary, whereas h at the same step can have
a two-dimensional Gaussian image density. Radius normalization changed the
measure's dimension, not the network map's rank.

T58's initial unnormalized two-input transport is unaffected by this caveat.
T67's fixed-target truncated-input transport also avoids this sphere projection.
Do not apply the Gaussian radial-independence derivation after conditioning on
an absolute-magnitude event such as a coordinate box: its radius and direction
are no longer independent. The correct truncated CDF formulation is separate.

For a future normalized forward solver, retain parametric sphere/ellipse patches
until their pushforward becomes a ball measure, or integrate them with exact
low-dimensional kernels. Simply depositing early mass as a smooth3D density
is an approximation requiring explicit refinement/error analysis.
