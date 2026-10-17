# T64: two analytic inner coordinates

## Question
Does the T55 triangle primitive make the existing H4T4 cone calculation cheaper without losing accuracy or its integration-only guarantee?

## Method
For signed-L1 rays R=[v0,v1,v2,v3], set Ai=u*v0+(1-u)*vi for i=1,2,3. The cone mass is |det R|/(2*pi^2) times the integral from zero to one of (1-u)^2 times the affine triangle integral of norm^-4. The triangle has barycentric coordinate measure 1/2; there is no extra factor two. Radial projection has Jacobian |det QR|/norm^3, leaving the degree-one spherical moment. This independently checks T55's measure and exponent.

Use the same geometry-only Gauss orders and Chebyshev bounds as ../stable_volume.py. Degree K=n-2 remains exactly integrated: the inner operator integrates a polynomial exactly and the remaining Jacobian has degree two. Triangle fallback quadrature bounds are added with positive outer weights. Geometry and roundoff remain outside these bounds.

## Status
Protocol frozen before T64 measurements. Root pushed claim 2d178fa. Canonical estimator remains unchanged. T55 owns primitive implementation/vectorization; this folder owns the outer driver and comparisons.

## Results
Pending. First controls: orthogonal triangle integral 3*pi/4; identity four-dimensional cone mass 1/16. Pilot fixed first32 original-model cones, then all three existing models if bounded cost permits. Existing fixed-GL20 references are comparisons, never allocation inputs. Noise is the previously fixed RNG51001, relative RMS scales0,.0025,.02.

## Compute accounting
Charge geometry, determinant/order planning, outer rules, endpoint construction, all analytic triangles and fallback quadrature, including guards. Report CPU and operation counters separately; no portable forward-equivalent claim and no new Monte Carlo. Batches at most512 cones; one numerical thread,2GB cap,120CPU-second cycle. Short independent controls may use the approved20CPU review lane; complete models use the main compute lock.

## Claims
[inferred] Two integrations can be eliminated analytically, reducing numerical dimension from three to one. [speculative] Batched triangles may reduce total cost. Only numerical quadrature, conditional on floating geometry, is claimed.

## Alternatives not ruled out
1. A missing measure factor or wrong exponent could create a consistent bias. The orthogonal triangle and identity-cone controls can reject this cheaply; full-model agreement with the prior positive3D rule tests the composition.
2. Thin-triangle cancellation could mimic convergence. Retain fallback counts and heuristic rejection diagnostics, compare against the positive3D rule, and keep rounding outside the analytic bounds.
3. Analytic overhead or frequent fallback may exceed savings. Freeze a first32-cone pilot and report total cost; do not select cones or models based on agreement.

## Failure modes
Floating geometry is uncertified; analytic moment guards do not bound rounding. Primitive source changes must be hashed. If fallback costs exhaust the cycle, stop explicitly and retain the negative. Outer bound plus fallback bound is integration-only. No global challenge or equal-compute sampling claim follows from these three cases.
