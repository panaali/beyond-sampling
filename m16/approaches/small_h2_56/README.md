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
