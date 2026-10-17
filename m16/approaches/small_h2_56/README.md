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
