# Independent interface and algebra controls

## T91 side allocation

Read interface_volume.py and its calls to mixed_volume.py. With n perpendicular
to the strictly positive input vector w, a both-active destination satisfies
n·h_new=n·(W h_old+w x)=n·W h_old. Classifying the whole scalar input row by
preactivation@normal is therefore correct: no missing scalar-x threshold is
needed. A source AREA cell crossing n·W h_old=0 nevertheless contains a
step discontinuity in the side-mass integrand. Exact scalar integration cannot
fix ordinary quadrature across that source-coordinate event.

The two helper states retain disjoint upper/lower area masses while the third
helper retains axes/ray; combined area and total-mass assertions are consistent.
Per-cell arrays are projected onto clipped side polygons, so inaccurate side
allocation can be exposed even when total area errors previously cancelled.
Quadrature atoms exactly on the preimage hyperplane are assigned to plus by
>=0; their positive numerical weight is an artifact because the true area
source assigns the hyperplane zero mass. The floating_zero_side_mass counter
is relevant, but near-boundary unresolved cells can err without exact zeros.

A sharp independent third-step check uses output rows[+n,-n]. First-index ties
send the origin and input ray to class0. Therefore

    area_plus = output0 - origin - input_ray - axis1,
    area_minus = output1 - axis0.

The sign conventions assume n=(-w2,w1), w>0. If the independent third-step
reference uses unbounded Gaussians and transport uses cutoffL6, allow the
additional1-(Phi6-Phi(-6))³, about5.9e-9. Splitting source polygons at nW h=0
is a distinct falsifier, with a triangulation-only control to separate event
alignment from simply changing quadrature layout. This source review does
not explain the measured bias quantitatively or establish that it is the
only remaining error.

## T39 antisymmetric offset fits

For Delta_ij=h(M at i,S at j)-h(S at i,M at j), swapping i,j gives
Delta_ji=-Delta_ij by construction. Consider equally weighted least squares
Delta_ij≈F_i+G_j on all ordered i≠j, with F0=G0=0. The objective is invariant
under(F,G)→(-G,-F). The anchored design has full column rank for n>=2: a null
fit evaluated on(i,0) forces F_i=0 and on(0,j) forces G_j=0. Thus its unique
minimizer necessarily has G=-F, regardless of the data's separability.

Writing r_i=sum_j Delta_ij (Delta_ii=0), the explicit fit is
F_i=(r_i-r_0)/n and G_i=-F_i. Substitution into the normal equations verifies
this independently. Exact separability requires every triangle circulation
Delta_ij+Delta_jk+Delta_ki=0. Fitted antisymmetry alone is tautological; R² and
nonzero cyclic residuals remain empirical information. Asymmetric pair
weights, missing directed pairs, or asymmetric regularization can invalidate
the invariance argument; numerical least-squares errors need separate care.
No source-data evaluation was required for this algebra-only review.

## T20 narrow Gaussian intervals

For real endpoints a<b, delta=b-a and exact midpoint m, midpoint quadrature
satisfies

    |integral_a^b phi(x)dx - delta*phi(m)| <= delta³*phi(0)/24.

Indeed phi''(x)=(x²-1)phi(x), whose maximum absolute value is phi(0); the other
stationary extrema at±sqrt3 have magnitude2 exp(-3/2)phi(0)<phi(0).

Crucial adjacent-float caveat: the exact midpoint commonly is unrepresentable.
If code instead evaluates at rounded mhat, the error also contains at most

    delta*|mhat-m|*sup_[a,b]|phi'|,

or the global bound delta*|mhat-m|*phi(1). This is orderdelta², potentially much
larger than the cubic quadrature term. Floating density, multiplication and
endpoint subtraction errors remain separate from both mathematical terms.

An algebraic alternative avoids materializing the midpoint:

    delta*phi(a)*exp(-a*delta/2-delta²/8).

In exact arithmetic this is delta*phi(a+delta/2), so the cubic quadrature
remainder applies to the expression directly. This is not a directed floating
enclosure; its elementary operation errors still need qualification. Neither
positive thin-mass evaluation nor this bound repairs unrelated gate/root/tie
logic. The requested sign/tie semantics remain exact, without an arbitrary
positive-gap tolerance.
