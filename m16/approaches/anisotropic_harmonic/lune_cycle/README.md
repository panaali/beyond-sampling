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
