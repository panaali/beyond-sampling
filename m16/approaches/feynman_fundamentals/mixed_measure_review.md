# T58 mixed-measure kernel review

Read-only review of mixed_volume.py, SHA256
f516415af1b2db03351bd5a5dbee54c120bc74a9267e400a0406eb885eff2f85,
and saved fourth_step_results.json, 2026-10-17. No numerical run was needed.
The reviewed prototype assumes both input weights positive, invertible W,
and nonzero determinants of [W*d,w] for the two axes and the input ray.

## Independent transfer formulas

For interior area density f and y>0, the change of variables at fixed fresh
input x gives

    f_new(y) = 1/|det W| integral phi(x) f(W^-1(y-w*x)) dx.

The integral is restricted to the source domain and x in [-L,L]. For a
piecewise constant rectangular source cell, these restrictions are scalar
interval intersections, so its contribution is an ordinary Gaussian interval
mass times the cell density and 1/|det W|. This matches area_density.

For a line h=r*d, with density g(r) with respect to dr, let M=[W*d,w]. Then

    (r,x)=M^-1*y, contribution=g(r)*phi(x)/|det M|.

Here r is the chosen line parameter, not arclength. The implementation uses
input amplitude on h=w*r, consistently with ray_factor*phi(r). No extra ||w||
factor is needed in this convention. The nonsingularity assumption ensures
these non-origin source lines become interior area under the both-active gate.

An old origin of mass p contributes p*(Phi(L)-1/2) to the positive input ray
and p*(1/2-Phi(-L)) to the new origin. The new input-ray density is p*phi(r),
0<r<L. The input law is not renormalized. Every positive source row integrates
to Z=Phi(L)-Phi(-L), so total retained path mass is Z^t. The prototype's box
recurrence max(W,0)*box+L*w is a valid coordinate upper bound.

The ray source quadrature uses Gaussian point values but rescales the two
weights in each cell to the exact cell mass. It is a positive mass-preserving
source approximation, not ordinary Gaussian quadrature with its usual
polynomial exactness. That distinction matters if a future bound is attached.

## Conservation is not a discretization certificate

The toy w=(1,1), W=diag(1,2), h0=0 obeys h2>=h1 at every step. Indeed,

    max(2*h2+x,0) >= max(h1+x,0)

whenever h2>=h1>=0. Thus positive mass on axis0 is impossible at every depth.
Saved fourth-step grid16 and grid32 outputs nevertheless have axis0 masses
2.55229724898e-5 and 4.17067553940e-7. Their total retained masses agree to
roundoff. The discrepancy is a falsifier for the inference that row conservation
alone validates the transported singular weights. Uniform rectangular cells
can leak source mass across the invariant boundary, even when the exact
transition from each quadrature node is integrated correctly.

A possible follow-up, assigned through the manager rather than implemented
here, is projection onto cell intersections with a proven reachable cone.
For this toy that means clipping by h2>=h1. Any resulting polygon quadrature
must preserve its actual area and source mass; simply deleting leaked nodes
and renormalizing would change the represented law without an error account.

Saved fourth-step grid32 errors also distinguish terminal readout integration
from projecting once more and then reading the histogram. The toy's largest
step-integrated output error is2.168825933e-4, versus9.896497051e-3 after final
histogram projection; trained H2T8 is1.370351055e-4 versus6.784729792e-3.
These are marginal output distributions after four steps, not sequence-rank
accuracy estimates for T8. Retaining the integrated endpoint avoids one visible
projection error but does not remove errors inherited from earlier steps.

## Pointwise boundary convention

If (W^-1*w)_i=0 and a query has (W^-1*y)_i exactly on a cell boundary,
area_density's closed tests can accept both adjacent cells for the whole
integration line. The resulting pointwise density may sum two different cell
values. This changes only a null set in area integration, but a half-open cell
convention is appropriate for a pointwise API and deterministic evaluations
exactly on mesh lines. It is separate from the finite-volume support leakage.

No missing Jacobian was found in this review. Actual convergence/error bounds,
readout ties in floating geometry, and future augmentation by target rank
remain the engine owner's validation responsibilities.
