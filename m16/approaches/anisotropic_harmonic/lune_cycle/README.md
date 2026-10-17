# T56 exact initial-lune support and cost audit

## Question
Does protecting both initial half-order boundaries improve the supported
spectral sketch at the same81 real degrees of freedom? Claim342d3be was
pushed before implementation. Earlier hemisphere estimator remains unchanged.

## Method
Use the existing twelve half-order cells and their frame(a,b,c), where a is
the nearer order-wall inward normal, b is the tangent component of the gap
bisector, and c=a cross b. In this frame each root is the spherical lune
phi in[pi/3,pi/2], z=x dot c in[-1,1]. Surface probability is dz*dphi/(4pi),
so each root has mass1/12 and conditional measure dz/2*dphi/(pi/6).

Use orthonormal product basis sqrt(2n+1)P_n(z) times
exp(i*2pi*m*(phi-pi/3)/(pi/6)), n=0..N and m=-M..M. The constant coefficient
of root1 is exactly1, with whole-sphere mass c0/12. Unlike the previous
hemisphere basis, no projected bisector defines the initial root; both true
root walls are exact support boundaries. Conjugate symmetry leaves81 real
DOF at N8,M4. The coordinate orientation and allocation differ, so this is a
local-support representation test, not isolation of one numerical parameter.

For a gate normal(nx,ny,nz), accepted azimuths satisfy
nx*cos(phi)+ny*sin(phi) >= -nz*z/sqrt(1-z^2). Intersect each periodic allowed
arc with the fixed root window, and integrate Fourier mode differences
analytically. Integrate the remaining radial Legendre products with positive
one-dimensional quadrature, split at every cut/window topology transition.
Use paired normal complementation to preserve gate-mass partition.

Freeze N8,M4 and radial order64 for the first H2T3 base pilot; order128 is
validation only. No degree or coordinate choice from reference accuracy.
Root-wall, orthogonality, generic-cut complement/Hermitian/PSD, nonzero Fourier
entry, constant-readout, zero/duplicate readout and scaling controls precede
trained interpretation. Compare with saved same81-DOF hemisphere and corrected
isotropic estimates; their costs remain separately qualified.

## Alternative explanations and falsifiers
1. Projected initial bisector leakage is dominant: exact lune support should
   materially improve the fixed trained result while root controls stay exact.
2. Internal gate/readout projection dominates: protecting both root walls may
   leave substantial bias or worsen it. No general spectral impossibility.
3. Narrow-root angular basis or cut integrals are ill-conditioned: independent
   direct entry integration and radial-order doubling must agree before the
   model test is interpreted. No clipping eigenvalues or silently normalizing.
4. Better accuracy costs too much: count all branching, plane normalization,
   frame transforms, coefficient/basis setup and dense operations; record
   transcendental/rule kernels explicitly rather than assigning fictitious
   unit FLOPs. A named-kernel event ledger is not a portable-FLOP certificate.

Strongest frozen local rescue is exact root support itself; no full learned
support, adaptive degree or oracle truth fit. If a numerical endpoint problem
arises, derive a separately frozen transformation and retain the failure.

## Status
Protocol frozen; primitive construction next. No numerical result yet.

## Results
Pending; commands and all failures will be added after bounded cycles.

## Compute accounting
One thread,2GB RSS,<=110CPU cycles through shared compute lock. <=20CPU
controls use the review lock. No new MC/training, no persistent model cache.
Canonical source remains intact; all new driver/primitive/audit code is here.

## Claims
[inferred] Product basis is orthonormal for conditional lune measure and
root mass is1/12. [speculative] This reduces actual accuracy-estimation bias.
No accuracy certificate, sampling advantage or portable-FLOP claim.

## Alternatives not ruled out
Projection leakage at internal gates/readout, different local coordinates,
full gate-cell support, and error-weighted degree allocation. Earlier T56
scope and all its failures remain valid; this phase does not erase them.

## Failure modes
Floating cone classification, signed projected densities, endpoint quadrature,
near-parallel boundaries, large branching cost and cost of special functions.

## Primitive controls
T55 independently accepts lune measure/basis normalization, with the warning
to include every wrapped arc intersection and handle the transverse-zero
case explicitly. The primitive integrates z on sine-mapped panels between
window-edge/cut-tangency transitions, smoothing square-root endpoints.
`controls.py` instead integrates radial Legendre products analytically and
then performs direct adaptive azimuth integration; its three nontrivial
entries agree within5.00e-16. Root-wall identity, complement, Hermitian,
eigenvalue and order64/128 controls pass. This independent integration order
shares the mathematical basis, not the arc-construction code. Runtime .044CPU.

Initial implementation corrections retained in this log: the first control
launch had a misspelled NumPy legval import, corrected before numerical work.
The first passing numerical run emitted a divide-by-zero warning when an
extremely thin radial panel rounded a node to a pole. The explicit limiting
empty/full arc now handles those nodes and records their count; it does not
clip eigenvalues or renormalize masses. Controls still pass without warnings.

## First trained pilot
H2T3 at N8,M4 gives .409040821130398 (error-.00113044059647), versus the
saved hemisphere error-.0022035339. Order128 changes1.31e-11. Zero/identical
readouts and positive coordinate scaling pass within1.12e-16. Geometry has69
leaves; the root and gate partition masses remain1. The observed error is
roughly halved but Q64 CPU .136 seconds exceeds the previous .033-second
hemisphere run; this is not an efficiency victory or matched-session result.

The expanded audit records147.5 million scalar arithmetic upper operations,
21,184 radial nodes,508,416 complex exponential and sine calls each, plus
explicit Gauss-rule and geometry elementary-function kernels. It includes
conservative bounds for both possible geometry branches, root frame setup,
coefficient complement, basis recurrence and all model matrix products.
Python bit-mask work is separate. This is coverage of numerical work with
named opaque kernels, not an accepted portable-FLOP total; transcendental and
rule setup costs are not silently assigned unit cost. Source geometry is
shared with the old sketch, not independently implemented geometry.

T55 code review accepts arc orientation, wrapped intervals, Fourier phase,
complement, wall/tangency knots and sine-map factor dz/2. It flags floating
normal/knot roundoff and confirms the scoped nature of exact initial support.

Next frozen cost phase: preserve this implementation as the default. A
separate algebraically equivalent mode skips wraps with no intersected arc
and forms each complex Gram matrix through two real matrix products. Count
both products and complex assembly explicitly. Compare matrices and full H2
outputs before a cold matched-session H2/H4 base comparison with hemisphere;
use N8,M4,Q64, with Q128 validation for the new H4 case. This is implementation
cost reduction at fixed numerical rule, not selection of lower resolution or
truth-tuned degrees. If equivalence fails, retain the failure and do not call
it an optimized estimator. No noisy sweep until this phase is assessed.

## Fixed-rule cost comparison
Optimized matrices agree within floating roundoff, and both full-model Q64
estimates agree exactly with the original lune implementation. Cold estimates
in the same process (seconds):

| model | hemisphere error / CPU | lune error / CPU | optimized lune CPU |
|---|---:|---:|---:|
| H2T3 | -.00220353 / .0320 | -.00113044 / .1793 | .1154 |
| H4T3 | -.00819957 / .1078 | +.00232448 / .5277 | .4313 |

H4 order128 changes2.75e-8, far below its residual. H2 order128 changes1.31e-11.
The fixed-rule optimization reduces measured CPU in this run and removes
unnecessary complex arithmetic/transcendentals; timing noise is visible
(order128 is slightly faster than64 here). Do not infer monotonic time or a
portable efficiency certificate. Initial support protection improves both
base errors, while leaving substantial projection bias and added cost.

Next frozen validation: use precisely the prior nine H2 parameter cases
(base plus8 absolute .05 noise,seed47001) and prior eight H4 .05 copies(seed0),
with existing references. Each estimate rebuilds its own state. Compare cold
hemisphere N8,M4 versus optimized exact-lune N8,M4,Q64; Q128 for every copy
is a refinement diagnostic only. Record every case, model hash, CPU, all work
and the squared-error-times-CPU proxy; do not call it same-budget/FLOP success.
No case, order or degree selected from accuracy. No new MC. A mixed result
remains mixed; exact initial support does not rule out internal-gate leakage.
