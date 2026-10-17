# T71: three analytic simplex coordinates for T5

## Question
Can the first spherical moment on S3 remove three T5 simplex coordinates and improve complete cold accuracy estimation?

## Method
Root pushed claim b7209a1 before implementation. For four endpoint columns V in R5, thin QR gives V=QR and c=R^-T1. Radial projection gives dOmega3=|detR|*||Ry||^-4 dy and c dot omega=1/||Ry||. Thus the standard tetrahedron integral of ||Vy||^-5 equals ||c||/|detR| times its first spherical moment in direction c. On S3, Delta(u0)=-3u0, so that moment is minus one third the sum of outward conormal components times S2 facet areas. Each facet area has an elementary positive atan2 formula. T55 independently checked the identity, sign convention, Jacobian and lack of an extra factorial before implementation.

Prototype uses oriented face cofactors and positive solid angles, guarding rank and boundary cancellation. A positive original-domain fallback uses matched Jacobi quadrature and the previously checked I5 segment formula, with the independently reviewed generic positive Chebyshev tail. Analytic rounding and geometry remain uncertified. T5 only; no T7/T9 work until demonstrated.

## Frozen initial protocol and alternatives
First test identity endpoints (truth2*pi/3), equal endpoints (norm^-5/6), collinear endpoints, deterministic signed-L1 endpoints, thin/asymmetric endpoints, orientation permutations and positive scaling. Compare ordinary cases to positive original-domain quadrature; preserve failed cases and count every analytic attempt/fallback. Before trained runs, verify the one-outer-coordinate Jacobian with constant integrands and the identity-ray Gaussian orthant1/32. Then compare the full T5 base and existing fixed .0025/.02 noisy references; no new Monte Carlo. Charge cold geometry, planning, primitive setup, elementary functions, fallbacks and outer integration.

Alternative1: a sign/factor error can reproduce selected shapes accidentally; exact identity/equal controls and permutation invariance can falsify it. Alternative2: facet cancellation may erase the expected savings on thin trained cones; measure all rejection/fallback counts and compare against positive quadrature. Alternative3: per-call QR/cofactor setup may dominate despite fewer coordinates; record full cold CPU and scalar component costs, without equating wall speed with audited FPE.

## Status
Initial primitive implementation and falsifiable controls pending. No solved or equal-compute claim.

## Results
Pending.

## Compute accounting
Every numerical cycle acquires /workspace/work/codex-compute.lock, uses one thread and ulimit -v2500000, at most110CPU seconds, and node blocks<=65536. Explicitly authorized review work may use the separate review lock for<=20CPU seconds. Peak memory<=2GB. Initial controls use the compute lock. Total FPE remains unaudited until geometry/library/setup costs are counted.

## Claims
[inferred] The spherical divergence identity removes three coordinates in exact arithmetic. [speculative] A stable implementation offers practical savings on all three trained T5 cases.

## Alternatives not ruled out
Conditioning, boundary cancellation, fallback-dominated cost and floating geometry errors.

## Failure modes
Rank-deficient tetrahedra and nearly cancelling facet sums require positive fallback. Tight quadrature bounds do not certify floating arithmetic. Shared signed-L1 planes are used only after checking their residual; incorrect normals cannot silently enter the formula.
