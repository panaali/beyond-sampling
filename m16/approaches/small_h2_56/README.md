# T48: H2,T5 and H2,T6 deductive accuracy

## Question
Can a complete weight-derived Gaussian cone calculation estimate trained H2 networks' accuracy competitively with sampling?

## Method
Recovery protocol, frozen before new experiments. Use full-trained AlgZoo seed0 for T5/T6, then fixed relative RMS perturbations .0025/.02, NumPy noise seed1. Earlier unpushed code was lost during a runtime restart. Prior reported numbers are historical, not regenerated evidence.

Cycle1 reconstructs bounded signed-coordinate-orthant gate enumeration and correct-answer cones. Normalize all rays to signed L1=1. Triangulate cross-sections and integrate the smooth simplex density using streamed tensor Gauss-Legendre rules. Cap each numerical cycle at120 CPU seconds; one thread, shared flock, <=2GB/process, chunks<=65536. T6 geometric degeneracy must fail explicitly. Initial T5 orders4 and8, followed by bounded adaptive refinement selected only from analytic error bounds. Validation uses seed48001 Gaussian inputs in chunks<=65536; no fitting to validation.

Two alternatives and cheapest checks: (1) apparent small-region success may hide missing or duplicated mass: integrate the full gate partition and compare correct-region results with independent direct validation; include all-tied outputs. (2) inverse-covariance probabilities may fail for slivercones: use positive simplex integration, independently verify analytic orthants, and report geometric uncertainty separately from quadrature error. (3) high-dimensional integration may cost more than analytic two-input tails plus lower-dimensional outer integration: compare operation counts and numerical refinement, label outer input evaluation as quadrature/sampling baseline.

## Status
Running recovery cycle. No solved claim; root handles independent review and integration.

## Results
Pending regenerated evidence. Historical pre-restart report: T5 258 gate cells,336 correctpieces,5015simplices; T6 759gatecells before Qhull degeneracy. These counts need reproduction.

## Compute accounting
All setup, clipping, hull calls, determinant/matrix operations and quadrature nodes must be charged. Validation forwards separate. No accepted cost advantage yet. Numerical runs acquire /workspace/work/codex-compute.lock, set OMP/OPENBLAS/MKL=1 and ulimit -v2500000.

## Claims
[inferred] Weight-derived cones permit deterministic probability integration. [speculative] T5 cone representation is economical. No independent full-estimator certification.

## Alternatives not ruled out
Floating-point geometry errors, noisy weights increasing complexity, expensive integration, hidden cancellation in output margins.

## Failure modes
Positive-volume logit ties require first-index argmax. Tiny cells cannot be silently discarded. Analytic quadrature bounds do not certify floating-point polytope construction.

## Integration identity and error bound
For same-orthant rays V with signed L1 norm1, dimension d and p=d/2, mass is Gamma(p)|detV|/(2pi^p) times the standard-simplex integral of ||Vy||^-d. Set a=max(1/d,min(V.T@V)), b=max(diag(V.T@V)), m=(a+b)/2, rho=(b-a)/(b+a). Tensor GL order L under the Duffy map integrates the Taylor polynomial in q=||Vy||² through K=floor((2L-d+1)/2), since q has degree2 per coordinate and the Jacobian degree at most d-2. If r=rho*(p+K+1)/(K+2)<1, uniform remainder B=m^-p*(p)_(K+1)/(K+1)!*rho^(K+1)/(1-r). Probability error <=Gamma(p)|detV|/(2pi^p)*2B/(d-1)!. This algebra was independently checked by the T47 agent before restart. It assumes correct geometry and exact arithmetic; implementation remains to verify. An upper bound for any whole simplex is d^(d/2)Gamma(d/2)|detV|/[2pi^(d/2)(d-1)!].

## Recovery cycle1 results
Reproduced T5 258 gate cells and336 correct pieces. Keeping every nonzero determinant produces5870 simplices (historical5015 used a1e-14 determinant cutoff). Positive simplex GL4/GL8 estimates are0.2792704228447156 and0.27923837663663276; GL8 full gate partition mass1.0000048414054903. The current Taylor error bounds are far too loose (0.488 for correct mass), so this is not a certified estimate. Independent validation131072 inputs gives0.2810516357421875±0.001241614985096441 standard error; discrepancy1.46SE. T6 reproduces759 gate cells and fails explicitly on clipping degeneracy.

`cycle1.py` reproduces these results; `cycle1.json` records timings and primitive checks. `t5_cones.npz` and `t5_partition.npz` cache derived geometry only, not fitted advice; all setup must be charged if used. Next cycle improves the analytic quadrature bound via a Bernstein ellipse and adaptively refines only using geometric error bounds. No validation-based selection.

## Recovery cycle2 results
`cycle2.py` uses a sharper Bernstein-ellipse bound. For q=m+delta*x and1<R<(sqrt(b)+sqrt(a))/(sqrt(b)-sqrt(a)), Re(q)>=m-delta*(R+1/R)/2>0 on the ellipse. Set M to this lower bound raised to -p. Chebyshev coefficients imply degreeK truncation error<=2M*R^-K/(R-1), hence probability quadrature error<=Gamma(p)|detV|/(2pi^p)*4M*R^-K/[(R-1)(d-1)!]. The T51 agent independently checked the algebra and normalization. A fixed32-point R grid optimizes only the analytic bound, not validation results.

T5 seed0 adaptive estimate **0.27923737135451504**, with conditional quadrature error bound **0.000024975576259353533**. Independent full-gate partition integration gives **1.000000000087144** with bound0.00002373337787157757. Orders4..28 were assigned from geometry and error bounds alone. Both integrations plus planning/checks took9.177 CPU seconds, peak RSS62MB. This bounds integration error conditional on the generated floating-point geometry; it does not bound clipping/triangulation error or certify challenge success.

Accuracy integral:170,340,864 quadrature nodes,10,390,792,704 conservative arithmetic-operation proxy excluding170,340,864 general-power calls and setup. Partition validation adds119,819,904 nodes and7,309,014,144 arithmetic proxy. Each matrix product costs2*d*d-d scalar multiply/adds, norm2*d-1, weighted accumulation2; the reported proxy2*d*d+2*d+1 is intentionally larger. Geometry, planning, ray determinants, node generation and power implementation still need audited costs. Wall speed is not evidence of efficiency versus sampling. No extra Monte Carlo was run in cycle2.

## Cycle3 protocol
Preserve the seed0 result and run the predeclared T5 noisy copies with the same geometric allocation tolerance1e-4, complete gate-partition validation, and the existing131072-input validation protocol. Replace generic powers by explicit square/multiply/divide operations and a single square root for odd dimension, giving an auditable operation inventory. Count geometry separately; do not invent a Qhull FLOP proxy.

T6 currently fails on degenerate clipping. Test a conservative geometric omission rule: if a Qhull failure occurs, project its vertices by dropping the last coordinate, rotate those coordinates with SVD, and enclose the projected polytope in a rectangular box. If its side lengths have product B, same-orthant normalization yields probability<=Gamma(d/2)d^(d/2)*B/(2pi^(d/2)). Only if this bound<=1e-12 may that ambiguous piece be omitted; explicitly count and sum the bound. Otherwise retain the exception. This is a geometric bound conditional on floating vertices, not an interval-arithmetic certificate. It does not justify silent discard or repair earlier geometry tolerances. First recover T6 geometry and low-order integration only; defer costly refinement if the node plan is too large.

## Cycle3 results
The two noisy T5 copies enumerate successfully: noise.0025 gives406 gate cells,541 correct pieces,10940 correct simplices; noise.02 gives443 gate cells,558 correct pieces,11490 correct simplices. Full gate-partition probabilities are1.000000000024983 and1.000000000044469, with conditional quadrature bounds2.86e-5 and2.69e-5. Accuracy plans375,350,784 and333,953,360 nodes exceeded the initial300M per-integral guard and were explicitly skipped; next cycle runs those separately within the120CPU-second cap. Validation reproduces0.3120422363 and0.3266830444, SE0.00128/0.00130. Cycle3 took19.76CPU seconds and250MB peakRSS.

The harsh copy required one explicit ambiguous-clip omission with mass upper bound9.17e-22. T6's ambiguous clip has box bound3.887e-4, so it is NOT discarded; further preconditioning is needed. The geometry remains numerically conditional. Root review identified two additional omission paths (rounded vertex counts and zero determinants); these are being changed to use the same explicit bound rather than silent dropping.

## Geometry review corrections
Rounded-vertex collapse and computed-zero determinants now require explicit bounding-box omission accounting, just like failed clipping. Plane normalization first divides by the largest absolute component, avoiding ordinary norm underflow for very small positive weight rescalings. These changes were requested by root review.

The T6 failure is a Qhull wide-facet merge error, not a genuinely low-rank polytope. A saved failing22-vertex example has projected singular values2.057,1.058,.472,.168,.0181; Qhull misses a point by8e-6 in its raw coordinates, while an invertible SVD whitening succeeds. Geometry now retries failures through that affine preconditioner only if the singular-value ratio exceeds1e-12, checks all transformed vertices against all output facets (residual<=1e-10), and counts each retry. No random jitter is added. The original numerical geometry caveat remains. Upcoming full-partition checks will test this repair.

## Cycle4 results
Completed both T5 noisy accuracy integrals with the revised geometry: .0025 gives0.31063601436620786 with conditional quadrature bound2.32966e-5; .02 gives0.3250610824408921 with bound2.24711e-5. Corresponding accumulated numerical-geometry omission bounds are3.83e-14 and2.81e-14. Validation discrepancies are1.10 and1.25 standard errors. Gate affine-map checks at every cell centroid differ from direct recurrence by at most2.23e-16. Arithmetic inventories and all bounds are in `cycle4.json`.

Affine preconditioning allows T6 geometry to complete:759 gate cells,1330 correct pieces,136438 simplices,3 whitening retries, explicit omitted-mass upper bound1.41e-13. Low-order GL4 gives accuracy0.245247 and full-gate mass0.982905, with huge integration bounds9.75 and39.96; this is an insufficient numerical rule, not a solved T6 estimate. Geometry took7.22CPU seconds; full cycle41.82CPU seconds,363MB peakRSS. Refinement must separate integration error from remaining geometric error.

## Cycle5 protocol: analytically eliminate one integration coordinate
For T5, integrate the last Duffy coordinate exactly. For segment X(u)=(1-u)A+uB, set r=||A||,s=||B||,D=A dot B,S=rs+D. Then integral ||X(u)||^-5 du from0to1 equals

`(r+s)*[(r*r+s*s-r*s)*S+(r+s)**2*r*s]/[3*r**3*s**3*S**2]`.

Differentiate the known inverse-cube integral (r+s)/(rsS) with respect to the common quadratic constant to derive this formula. Two sibling agents independently checked the algebra, A=B and orthogonal-unit cases. Same-orthant rays give S>=1/d, avoiding antipodal singularities. The final Duffy Jacobian exponent is zero, and exact last-coordinate integration preserves the Taylor/Chebyshev polynomial degree in all remaining variables; the existing analytic quadrature bound therefore survives. This reduces T5 tensor nodes from n^4 to n^3. Before using it on trained cones, validate arbitrary deterministic endpoints against high-order positive one-dimensional quadrature and analytic cases, then reproduce all three previously committed T5 estimates and the full partition. Preserve all discrepancies and count actual scalar operations separately from Qhull setup. No model-specific tuning.

## Cycle5 results: one coordinate eliminated analytically
The positive segment formula passes15 analytic/deterministic endpoint checks against128-node independent positive one-dimensional quadrature, including equal, orthogonal, collinear and nearly equal endpoints. Algebra and inherited-error-bound reasoning were independently checked by root and two sibling agents before implementation.

Using exactly the prior frozen per-cone orders and bounds, the three T5 estimates are:

| relative noise | accuracy | conditional quadrature bound | numerical nodes | arithmetic ops excluding sqrt/setup | square roots |
|---|---:|---:|---:|---:|---:|
|0|0.2792373713559676|2.49756e-5|9,465,648|851,925,929|18,931,296|
|.0025|0.31063601436566657|2.32966e-5|20,237,216|1,821,382,349|40,474,432|
|.02|0.32506108244184867|2.24711e-5|18,708,560|1,683,804,857|37,417,120|

All differ from the previous full-dimensional positive quadrature by less than1.5e-12. The original full-gate partition gives1.0000000000928302. This cuts numerical node counts by18–20x while preserving the analytic integration bound. Each node uses90 scalar arithmetic operations plus2 square roots; explicit scalar per-simplex aggregation is added. Determinant, planning, node generation and Qhull setup are still separate and require review before claiming an equal-compute win. Planning plus integration took1.49,2.93,2.76CPU seconds respectively, excluding geometry already reported. `cycle5.json` contains complete counts and checks; `cycle5.py` reproduces them. No new random inputs were used in this cycle.

## Usable API and next validation cycle
`estimator.estimate(model,tolerance=1e-4)` now performs the complete T5 calculation and returns its estimate, conditional quadrature bound, explicit geometric omission bound, detailed operation inventory and timing. It detects the exact constant-first-index classifier case from componentwise output-row dominance. Its provisional `est(model,budget,rng)` adapter returns infinite total cost for ordinary enumerated models until geometry/setup accounting is audited; it never passes known integration arithmetic off as total cost. The fixed-tolerance calculation does not enforce an arbitrary total budget, and must not be submitted as an equal-compute leaderboard entry yet. RNG is unused.

Next validation cycle is frozen: run the same T5 method on componentwise-dominant/identical-output cases with known accuracy1/5; check the enumeration engine independently on a constant-readout model; repeat seed0 after independently scaling input/output weights by1e-16, compare derived estimates within reported bounds; investigate T6 full-partition refinement before making accuracy claims. Additional training seeds, if available, will use the same frozen tolerance and no outcome-based selection.

## Cycle6: tie/scaling checks and T6 pruning
The exact constant-first-index shortcut returns1/5. Bypassing that shortcut and integrating its full gate/rank geometry independently gives0.200000000040757, absolute error4.08e-11 within the2.16e-5 quadrature bound. Scaling Wi or Wo independently by1e-16 changes the trained seed0 estimate by only5.14e-13. No random inputs were used.

For T6, use the whole-simplex upper bound Gamma(d/2)|detV|*a^(-d/2)/[2pi^(d/2)(d-1)!], a=max(1/d,minGram). Of136438 simplices,73853 have upper bound<=1e-4/136438. Omitting them costs an explicitly summed mass bound5.00327e-6, leaving62585 cones. This is a controlled probability omission conditional on geometry/arithmetic, not a numerical determinant cutoff. `cycle6.json` and `t6_retained_cones.npz` preserve the counts and retained geometry.

## Cross-seed protocol
Root approved all five full-training T5 seeds, same tolerance1e-4, identical code and no outcome-based selection. Seed0 is already evaluated. Next run seeds1,2 then3,4 in bounded cycles<=120 CPU seconds, preserve every estimate and failure, and record checkpoint hashes. New-model validation uses131072 Gaussian inputs (seed48001, chunks65536); no additional MC is run on existing cases. Full gate-partition checks accompany new seeds. T6 work uses T55's separately validated inverse-sixth segment primitive, retaining its series-error/roundoff qualifications.

## Cross-seed replication, first batch
Fixed-protocol training seeds1 and2 complete. Seed1 accuracy0.27920628913235723 with quadrature bound2.62342e-5; full partition1.0000000000998137; validation discrepancy1.58SE. Seed2 accuracy0.3047754587821175 with bound2.24263e-5; full partition1.0000000000157727; validation discrepancy0.36SE. Geometry omissions are bounded by1.65e-14 and2.91e-13. Accuracy integrals use10.45M and51.29M numerical nodes respectively, showing real cross-seed cost variability. Both full geometry and full partition costs are retained in `crossseed_1_2.json`; batch CPU25.19 seconds,249MB peakRSS. No seed was selected by accuracy or cost. Seeds3,4 are the next already-frozen batch.

Reproduce: `crossseed.py 1 2` or `crossseed.py 3 4`, with the shared-lock, single-thread and memory-limit command used for earlier cycles. Checkpoint SHA256 values are recorded per seed.

## Cross-seed replication complete
All five predeclared T5 training seeds now complete with the same method and tolerance. Seed3 accuracy0.30563489736028726, bound2.09377e-5, partition1.0000000000939906, validation discrepancy1.23SE. Seed4 accuracy0.24362933860588015, bound2.25304e-5, partition1.000000000012262, validation discrepancy0.29SE. Their geometry omission bounds are4.49e-14 and3.21e-14. The second batch took10.89CPU seconds and237MB peakRSS.

Across all five seeds, accuracy varies24.36–30.56%, correct simplices5814–30708, numerical nodes9.47–51.29M, and every partition is within1.0e-10 of one. Every independent131072-input validation lies within1.59SE of the deductive estimate. This supports numerical reproducibility and robustness; it does not remove the conditional floating-geometry qualification or unaudited setup cost. Independent DD review is owned by root/T52.

## T6 next refinement protocol
Use T55's independently validated inverse-sixth segment primitive (commit f1ed984) to remove the last Duffy coordinate. Compare fixed orders4,8 on the full gate partition before refining accuracy. Match remaining Duffy Jacobian factors with Gauss-Jacobi rules alpha=d-2-j,beta0; mapped weights already include(1-u)^alpha. Positive weighted quadrature then integrates q^K exactly throughK=n-1, so the same Bernstein bound uses that degree instead of the Legendre degree floor((2n-d+1)/2). Preserve the existing Legendre results as comparisons. Carry the primitive's series truncation separately; closed-form floating rounding remains unbounded, as elsewhere in this numerical calculation. Limit new runs to120CPU seconds and refuse an oversized plan rather than silently stop mid-sum.
