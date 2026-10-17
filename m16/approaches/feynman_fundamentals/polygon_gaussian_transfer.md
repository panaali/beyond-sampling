# Analytic Gaussian-CDF polygon transfers (T92 backlog derivation)

T92 was reserved as BLOCKED, backlog only, in TASKS-only push06680a4. No
implementation is claimed. Its use requires a concrete T91/T67 precision/cost
need; correcting side-event quadrature alone did not fix the T91 endpoint.

## Exact domain of the reduction

Let h range over a convex source polygon P with uniform or affine density
rho(h)>=0, and let scalar x have standard-normal density. In an H2 ReLU step
with strictly positive input vector w, a positive destination rectangle gives
scalar bounds

    lower(h)=max(-L,(lo0-(Wh)0)/w0,(lo1-(Wh)1)/w1),
    upper(h)=min( L,(hi0-(Wh)0)/w0,(hi1-(Wh)1)/w1).

For each active lower/upper pair, clip P by inequalities selecting that pair
and lower<=upper. There are at most nine pairs for these three-by-three bound
families. The resulting integrand is rho(h)[Phi(upper(h))-Phi(lower(h))].
All bounds are affine in h. Identically equal bound forms need first-index
ownership; otherwise separate active-pair cells could overlap with positive
area. Nonidentical ties are lines of zero area. Degenerate zero-area polygons
are not area measures and must remain separate axes/ray/atom representations.

Inactive ReLU coordinates replace a rectangle bound by x<=-(Wh)i/wi and give
analogous finite families; singular input weights or mixed signs require
rederiving inequality directions rather than applying this formula blindly.
For unbounded inputs an infinite bound contributes0 or the total polygon
mass and should not be passed through finite antiderivative expressions.

## Uniform polygon integral

Define

    J(t)=t Phi(t)+phi(t),
    K(t)=((t²+1)Phi(t)+t phi(t))/2.

Then J'=Phi and K'=J. For t=a·h+b and A=a·a>0, the vector field
F(h)=a J(t)/A has divergence Phi(t). For a CCW edge from v0 to v1, write
e=v1-v0, t0=a·v0+b, t1=a·v1+b. Its contribution to the polygon integral is

    (a·(e_y,-e_x))/A * average_0^1 J(t0+u(t1-t0)) du.

The average is [K(t1)-K(t0)]/(t1-t0), or J(t0) when the arguments are exactly
equal. No extra edge length or triangle factor belongs here: the unnormalized
outward normal already contains edge length. For a=0 the answer is simply
area(P)*Phi(b). Subtract the lower-CDF integral from the upper-CDF integral
and multiply by a constant density when applicable.

## Affine source density

For rho(h)=c+g·h, set beta=(a·g)/A. The field

    F(h)=a/A * [rho(h)J(t)-beta K(t)]

has divergence rho(h)Phi(t); the extra gradient term cancels exactly. Its
edge integrals require only K and

    L1(t)=[t³Phi(t)+(t²-1)phi(t)]/3,       L1'=t J(t),
    L2(t)=[(t³+3t)Phi(t)+(t²+2)phi(t)]/6, L2'=K(t).

Along an edge with distinct t0,t1, express rho=alpha+gamma t and use
[alpha DeltaK+gamma DeltaL1]/(t1-t0) for the first term and
DeltaL2/(t1-t0) for the second. If t is constant, integrate the linear rho
by its endpoint average. The a=0 case is Phi(b)*area(P)*rho(centroid).
Thus real-arithmetic source quadrature is unnecessary for this finite class.

## Why source event cuts often have a simple geometry

Comparing destination-coordinate input thresholds gives

    n·Wh = n·(destination grid corner),  n=(-w1,w0).

These are parallel source lines. The interface-side event n·Wh=0 is one of
them. Comparisons with input cutoffs instead give affine planes normal to a
row of W. Readout lines add analogous affine bounds. This explains why
source quadrature can cross a sharp event despite exact scalar-input CDFs;
it does not assert that all source-density features share this orientation.

## Required falsifiers before implementation or use

Alternative1: boundary orientation or normalization is wrong. Check constant
CDF gradients, a square/triangle with separable axis-normal gradient, exact
area/mass, coordinate rotations and reflected orientation, then compare with
independent positive original-domain quadrature. Check all-active-pair sums
against a one-piece constant transition and explicit identical-bound ties.

Alternative2: a correct identity is unstable or too expensive. Stress a→0,
t1≈t0, narrow upper/lower gaps, far Gaussian tails and thin polygons. Large
signed edge contributions and upper-minus-lower cancellation can dominate
the true positive mass. No clipping of negative mass is an error certificate.
Use stable divided differences, higher precision, or a positive quadrature
fallback with explicit error. Record all clipping, root comparisons, polygon
edges, CDF/density calls, reconstruction and fallback costs. A naive
source-cell×destination-cell implementation may be slower than current source
quadrature even though each integral has a finite formula.

This derivation is not an implementation, floating enclosure, efficiency win,
or claimed accuracy rescue. It can remove source-integration error for a
represented density; it cannot remove incorrect cell masses or density-shape
projection error. Explicit line-source components and cutoff interfaces remain
separate from the area reconstruction.
