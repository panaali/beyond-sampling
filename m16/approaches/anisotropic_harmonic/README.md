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

First primitive controls (`checks.py`, `checks_results.json`) complete in
.028CPU. Across N/M=(4/0),(4/1),(8/1),(16/2) and six fixed normal directions,
complements/Hermitian/eigenvalue checks pass; hemisphere intersection mass
error<=2.46e-10, and an independent direct-z adaptive integral agrees within
9.66e-15. Twelve root masses sum1 and an aligned protected-wall readout loses
zero mass. Doubling the positive quadrature rule changes a difficult near-
tangential cut matrix by up to3.38e-8, so primitive integration is numerical
and not certified. Preserve that limitation in trained comparisons.

First H2T3 pilot (`pilot.py`) uses the existing exact accuracy
.41017126172687207. Zero and identical readouts return1/3. Errors for
N/M=4/0,4/1,8/1,16/1,8/2 are respectively -.00956,-.02481,-.02426,-.02422,
-.01149. Every root/gate partition is conserved. Normal refinement at fixed
M1 stalls, while M2 changes the result substantially; the M0 near-hit is not
selected. Each estimate takes about.020-.024CPU, but dense arithmetic ranges
from.31M to13.0M explicitly counted ops. Indexing/comparisons and some setup
arithmetic are not fully audited, so that ledger is not a total-cost bound.

Next fixed discriminator: N8 with M4 and M8, plus N4/M2 for equal-state context;
repeat N8/M4 with doubled cut quadrature to isolate primitive error. Compare
with the independent T57 isotropic evaluator at D2/4/8 with first-index and
root-mass fixes, charging its cold setup and gate-map omissions. Actual CPU is
reported separately from each implementation's imperfect arithmetic ledger.
This tests tangential under-resolution as the strongest remaining cheap
rescue; it is not an unbounded degree sweep. No new random inputs or fitting.

Tangential rescue (`rescue.py`) improves N8 errors from-.02426(M1) to
-.01149(M2), -.00220353(M4), and-.00191416(M8). Doubling cut quadrature at
N8/M4 changes the estimate by9.4e-14, so primitive integration is not the
source of this remaining error on this model. Corrected isotropic errors at
D2/4/8 are-.01670,-.01611,-.00502076. N8/M4 and isotropic D8 both store81
coefficients; measured cold CPU is.0333 versus.0520 seconds, with absolute
error smaller by2.28x. This is a limited improvement on one model, not the
requested>10x robust fixed-cost gain. M8 costs.0665 seconds and improves only
slightly, so it is not automatically preferable. Arithmetic ledgers differ
and omit some work; no matched-FLOP claim is made.

Next finite controls frozen: test N4/M4 and N16/M4 to distinguish normal
resolution from the tangential bottleneck, without expanding M further.
Independently evaluate the fixed N8/M4 rule and corrected isotropic D8 on the
eight already-defined T47 absolute-noise.05 copies (RNG47001, original Wi/Wh/Wo
order), using their existing exact truths. No new input sampling or truth
calculation. These copies were not used to design this sketch. Retain every
copy and report squared-error times actual cold CPU, both each model and
aggregate; this compares implementations, not a portable operation benchmark.
Do not retune the fixed N8/M4 choice on these copies. Independent review must
check any apparent improvement and remaining alternatives before handoff.
