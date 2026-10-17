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
