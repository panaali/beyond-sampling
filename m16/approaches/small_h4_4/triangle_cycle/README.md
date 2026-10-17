# T64: two analytic inner coordinates

## Question
Does the T55 triangle primitive make the existing H4T4 cone calculation cheaper without losing accuracy or its integration-only guarantee?

## Method
For signed-L1 rays R=[v0,v1,v2,v3], set Ai=u*v0+(1-u)*vi for i=1,2,3. The cone mass is |det R|/(2*pi^2) times the integral from zero to one of (1-u)^2 times the affine triangle integral of norm^-4. The triangle has barycentric coordinate measure 1/2; there is no extra factor two. Radial projection has Jacobian |det QR|/norm^3, leaving the degree-one spherical moment. This independently checks T55's measure and exponent.

Use the same geometry-only Gauss orders and Chebyshev bounds as ../stable_volume.py. Degree K=n-2 remains exactly integrated: the inner operator integrates a polynomial exactly and the remaining Jacobian has degree two. Triangle fallback quadrature bounds are added with positive outer weights. Geometry and roundoff remain outside these bounds.

## Status
Done: full three-case validation and matched cost comparison passed. Protocol frozen before T64 measurements in54ef5fe. Root pushed claim2d178fa. Canonical estimator remains unchanged pending root/DRI review. T55 owns primitive implementation/vectorization; this folder owns the outer driver and comparisons.

## Results
Orthogonal triangle and identity cone controls pass to4.45e-16 and1.39e-17 absolute error. Fixed first32-cone pilot matches the previous3D rule to2.17e-19;468 triangles include6 guarded fallbacks. Its integration took.00405CPU versus.00797CPU for the previous rule.

Full results (full_results.json):

| noise | estimate | difference from fixed3D reference | outer+inner integration bound | total CPU | triangle integration CPU |
|---|---:|---:|---:|---:|---:|
|0|.477013125994119|2.78e-16|6.88653e-10|2.456|.572|
|.0025|.5014929079254983|1.11e-16|8.69052e-10|2.548|.532|
|.02|.3638126426226529|1.67e-16|5.98247e-10|1.924|.300|

Triangle counts127748/124861/93894; guarded fallback counts8226/10349/2474 and fallback nodes1341072/2243568/239296. Weighted fallback integration bounds1.77e-13/1.60e-13/2.22e-14 are included above. Previous3D node counts were30832112/26687215/18126648. Geometry remains the main cost. Neither node reduction nor historical CPU alone establishes portable speedup.

Run from the repository root with one thread and the shared compute lock:

```
/workspace/clean-neurons/.venv/bin/python m16/approaches/small_h4_4/triangle_cycle/run.py --triangle-dir /workspace/work/feynman_fundamentals/m16/approaches/feynman_fundamentals --geometry /workspace/clean-neurons/m16/approaches/small_h2_56/geometry.py --checkpoint /workspace/clean-neurons/work/checkpoints/H4_T4_seed0.pth --mode full
```

Use --mode pilot for the fixed sample. Source hashes are saved in both result files; T55 batch primitive is commit5d08526. Existing fixed-GL20 references are comparisons, never allocation inputs. Noise is the previously fixed RNG51001, relative RMS scales0,.0025,.02. Follow-up cost protocol: regenerate the same three geometries once each, run both integrators on the same rays, alternating execution order by case; report geometry plus each integration cost, with no repetitions chosen by outcome.

The matched-process check (cost.py, cost_results.json) took11.17CPU in total:

| noise | triangle integration CPU | previous integration CPU | triangle total CPU | previous total CPU | total speed ratio |
|---|---:|---:|---:|---:|---:|
|0|.648|1.075|2.818|3.246|1.152|
|.0025|.580|.952|2.700|3.072|1.138|
|.02|.288|.600|1.936|2.248|1.161|

Both totals include the same freshly measured geometry cost. Integration speed ratios are1.66,1.64,2.08; overall CPU savings are12–14%. Matched estimates differ by at most2.22e-16. This is one hardware-specific comparison with alternating execution order, not a confidence interval or portable operation-count proof. Reproduce using the same interpreter, one-thread settings and compute lock with `m16/approaches/small_h4_4/triangle_cycle/cost.py`.

## Compute accounting
Charge geometry, determinant/order planning, outer rules, endpoint construction, all analytic triangles and fallback quadrature, including guards. Report CPU and operation counters separately; no portable forward-equivalent claim and no new Monte Carlo. Batches at most512 cones; one numerical thread,2GB cap,120CPU-second cycle. Short independent controls may use the approved20CPU review lane; complete models use the main compute lock.

## Claims
[inferred] Two integrations can be eliminated analytically, reducing numerical dimension from three to one. [measured] The three unchanged test cases agree with the previous positive3D rule to2.8e-16; matched total CPU improves by factors1.14–1.16. Only numerical quadrature, conditional on floating geometry, is claimed. Root/DRI acceptance and portable cost validation remain separate.

## Alternatives not ruled out
1. A missing measure factor or wrong exponent could create a consistent bias. The orthogonal triangle and identity-cone controls can reject this cheaply; full-model agreement with the prior positive3D rule tests the composition.
2. Thin-triangle cancellation could mimic convergence. Retain fallback counts and heuristic rejection diagnostics, compare against the positive3D rule, and keep rounding outside the analytic bounds.
3. Analytic overhead or frequent fallback may exceed savings. Freeze a first32-cone pilot and report total cost; do not select cones or models based on agreement.

## Failure modes
Floating geometry is uncertified; analytic moment guards do not bound rounding. Primitive source changes must be hashed. If fallback costs exhaust the cycle, stop explicitly and retain the negative. Outer bound plus fallback bound is integration-only. No global challenge or equal-compute sampling claim follows from these three cases.
