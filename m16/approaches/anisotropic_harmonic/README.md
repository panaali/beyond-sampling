# T56: a harmonic sketch supported on the near tie hemisphere

## Question
Can high resolution normal to a true tie face and low tangential resolution
avoid the isotropic sketch's leakage penalty at comparable actual compute?
Root reserved T56 in pushed fc741c8. Claude owns T73/T75; T57 independently
reviews the original isotropic engine, which is not duplicated here.

## Method
First freeze a strongest plausible rescue, rather than only stretching the
old full-sphere basis. Split the six T3 order cells into their two halves,
according to which adjacent tie gap is smaller. For the nearer wall normal a,
use a basis supported exactly on a dot x>=0. Coordinates are
x=z*a+sqrt(1-z*z)*(b*cos(theta)+c*sin(theta)), z in[0,1]. The basis is
sqrt(2n+1) P_n(2z-1) exp(i*m*theta), n<=N, |m|<=M, orthonormal for the
CONDITIONAL hemisphere measure dz*dtheta/(2pi). Whole-sphere mass is half
the constant coefficient. High N resolves the tie-normal direction; small M
limits tangential work. Support never leaks across the chosen true wall.

Let g1=x_max-x_median and g2=x_median-x_min. In the half with g1<=g2,
a is g1's normal and the only additional root inequality is g2-g1>=0.
The other half exchanges them. Its bisector normal has cosine -sqrt(3)/2
with a, so every root lune has whole-sphere mass1/12. All12 roots must sum1.
Root coefficients come from one direct cut, not repeated projected cuts.

A generic cut matrix is the integral of conjugate(basis_i)*basis_j over
the hemisphere intersected with that halfspace. Integrate tangential Fourier
arcs analytically. Integrate the remaining z coordinate deterministically,
splitting at z=sqrt(1-(a dot v)^2) and substituting z=z_break*sin(phi) near
the arc endpoint. This numerical integration is not called exact. The
aligned-wall cut itself is exactly identity/zero on this supported basis.
Geometry may be shared read-only with existing DD code, but normalize every
nonzero plane before classification and handle identical-logit margins with
first-index argmax. Charge root setup, projections, gate-map multiplication,
geometry, and readout; no free model-dependent caches across estimates.

## Status
Protocol frozen before implementation/runs. T67 bounded phase is saved as
inconclusive; its independent review can be answered without changing T56.

## Results
None yet. Controls precede trained runs: hemisphere intersection mass,
complementary cuts, Hermitian matrices, bounded eigenvalues, root partition,
constant and duplicate readouts, and independent quadrature refinement.
The first trained case will be H2T3 with existing exact reference, at small
N=4/8/16 and M=0/1/2 under fixed resource caps, not a large sweep. Report the
full error/cost curve and comparisons using actual counted work.

## Compute accounting
One OMP/OPENBLAS/MKL thread, <=2GB RSS, virtual memory cap2500000KB. Numeric
cycles<=110CPU seconds; <=20CPU controls may use codex-review.lock, longer
work uses codex-compute.lock. No new Monte Carlo or training in this pilot.
Include complex arithmetic and rule/matrix setup, not just coefficient count.

## Claims
[speculative] Exact support at the nearest true tie wall can eliminate a
specific leakage mechanism. It does not imply full accuracy or competitiveness.

## Alternatives not ruled out
1. Root mass/tie bugs caused the original negative, not isotropic leakage.
T57 found both issues but their correction did not rescue trained cases.
Our root partition and first-index controls must pass independently.
2. Tangential variation or gate-boundary projection, rather than tie-normal
resolution, dominates. Compare N and M separately; retain nonmonotonic errors.
3. Supported roots merely shift leakage to the bisector or interior gates.
Inspect full mass conservation and readout error on the smallest exact case.
4. Cut integration or dense matrix cost overwhelms any approximation gain.
Refine the primitive and count all construction/applications before a win.

## Failure modes
No general spatial/spectral truncation certificate. The angular basis is only
L2 at the poles; ordinary polynomial exactness is not assumed. Numerical
cone geometry can misclassify nearly degenerate cuts, even with normalized
planes. Signed projected densities and apparent error cancellation remain
possible. Exact region branching may itself become prohibitive at larger T.
A failure here rejects only this finite formulation, not every anisotropic
or locally supported sketch.
