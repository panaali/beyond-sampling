# T51: H4T4 angular reduction

## Question
Can the user-prioritized four-neuron, four-input model's accuracy be calculated efficiently through a one-dimensional integral?

## Method
The T55 weighted two-input tail integrates the last two Gaussian inputs and the independent chi_2 prefix radius by spherical polygon first moments. The remaining normalized two-input prefix is a uniform angle. `estimator.py` applies Gauss-Legendre integration separately to the eight angular octants. This is deterministic outer numerical integration, not certified exact accuracy or an accepted strict mechanistic solution. It reuses T55 `tail.py` without edits.

`stable_volume.py` preserves a distinct, analytically derived four-dimensional simplex quadrature primitive from before the restart. It is not used by this experiment and has not yet passed executable checks.

## Status
Running, under existing pushed T51 claim. Original worktree was lost during runtime restart; earlier Plackett cone estimates are historical and unvalidated. Recovery protocol frozen before new measurements.

## Results
Cycle 1 predeclared: trained H4T4 seed0 and independent relative-RMS noise .0025/.02 using RNG51001. Angular GL orders2,4,8 per octant; refinement16 if total CPU remains below120 seconds. Independent Gaussian validation uses262144 inputs, seed51002, chunks<=65536. Save each completed model/order immediately. Compare refinement differences and Monte Carlo uncertainty, never treat agreement with noisy truth as proof. Check zero-logit tie ownership and input/readout positive scaling on bounded controls. All numerical runs take `/workspace/work/codex-compute.lock` and use one BLAS thread and a2GB memory limit.

## Compute accounting
Record CPU/wall time, angular nodes, deterministic prefix steps, clipping calls, affine products, moment calls and fallback counts. Sampling baseline includes generated inputs, forward passes and target/prediction argmax. Hardware CPU equivalents include geometry and moments; scalar primitive accounting remains incomplete and is not replaced by counting only network forward calls. Shared arc_eval is hard-coded T10, so this local dimension-generic runner preserves tuple interface and noise convention but cannot directly use its evaluator.

## Claims
[speculative] Analytic tail integration may make the remaining one-angle function cheap enough for useful accuracy estimates.

## Alternatives not ruled out
1. Narrow angular error features can escape low-order rules: cheapest test successive GL orders and an offset partition.
2. Apparent agreement may come from noisy ground truth or cancellation: compare independent sampled uncertainty and each noisy copy separately, then independent geometry review.
3. The conditional calculation may cost more than sampling: compare complete CPU time, not only nodes/model evaluations.

## Failure modes
Outer quadrature has unbounded discretization bias without further analysis; geometry uses floating tolerances; conditional zero-probability prefix ties require special handling if nodes land exactly on them; estimator budget is advisory and reported work may exceed it. No extrapolation to H16T10 or other seeds.

### Recovery cycle 1 partial result
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python m16/approaches/small_h4_4/run.py /path/H4_T4_seed0.pth` (run under the shared flock and memory cap).
Original GL2/4/8 estimates:0.4731219478/0.4772917066/0.4768521781, costing1.03/2.01/4.26 CPU seconds. Independent262144-input validation:0.4770965576 with standard error0.0009755374. The GL4-to-GL8 change0.0004395 exceeds the equal-CPU sampling standard error; convergence and efficiency remain unresolved. These data do not establish a win.

The .0025 noisy model completed GL2 (0.4978691891) but GL4 raised the T55 geometric moment bound guard. `moment_failure.json` preserves the offending polygon; `capture_failure.py` reproduces it. No clamping or silent omission was used. The T55 owner has been notified and the noisy/refinement cycle awaits its stability correction. Partial results survive in `results.json`.

### Recovery cycle 2
`python m16/approaches/small_h4_4/refine.py` under the same lock/thread/memory settings reproduces `refinement.json`. Original GL16=0.4771704463; shifting all eight partition boundaries by pi/16 at GL8 gives0.4722998837. This large sensitivity rules out trusting simple low-order angular rules. Wi and Wo scaling by1e-16 preserve GL2 within2.3e-16. The zero-logit control gives0.2500538788 at GL2, compared with exact0.25: outer quadrature itself has measurable bias. The preserved stable-volume primitive passed identity-orthant and split-orthant controls at GL8/16/20 within its stated conditional quadrature bounds; no cone enumerator was rebuilt.

### Cycle 3 protocol (frozen before measurements)
Hypothesis: known prefix-rank and ReLU transitions cause conditional probability discontinuities or narrow angular features. Alternative1: errors instead arise inside smooth prefix regions; test GL2/4/8 refinement after splitting at analytically derived prefix gate roots. Alternative2: positive-mass output ties create additional unrecognized transitions; preserve all partition/moment guards and compare one further subdivision of each interval. Build first-two-step gate boundaries directly from weights: cos(theta)=0 fixes first-step gates; second-step preactivation is a*cos(theta)+Wi*sin(theta), where a=Wh@(Wi*first_gate). Include all valid roots, eight original octant boundaries and prefix-rank ties. No fitted knots or sampled accuracy are used. Test original before noisy copies; keep old data and classification as numerical outer integration.
