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

### Cycle 3 result
`python m16/approaches/small_h4_4/prefix_run.py` gives original estimates0.4750663544/0.4768213989/0.4768493001 at GL2/4/8 on analytically split prefix intervals, then0.4769827522 after bisecting each interval at GL8. The subdivision changes the result by0.00013345, so prefix splitting helps isolate known transitions but does not by itself establish convergence. No guard failed in this cycle; see `prefix_results.json` for full timings and node values.

### Cycle 4 protocol
Use the T55 positive fallback (upstream86c3759) and retain all fallback counts, nodes, and the outer-weighted sum of its quadrature bounds. Those bounds do not cover outer discretization or floating geometry. Run unchanged prefix-split GL4, GL8 and subdivided GL8 on original plus the two predeclared noisy models. Full-cycle CPU cap remains115 seconds; each completed rule is saved immediately. No rule is selected using sampled accuracy.

### Cycle 4 result
`python m16/approaches/small_h4_4/run.py /path/H4_T4_seed0.pth --mode prefix --rules 4:1,8:1,8:2 --output prefix_noise_results.json` under the common lock/thread/memory settings completed in73.30 CPU seconds. Final subdivided-GL8 estimates are0.4769827522/0.5014803932/0.3638047717 for noise0/.0025/.02, costing13.82/14.56/13.56 seconds per estimate. The GL8-to-subdivided-GL8 changes are+0.00013345/+0.00015422/-0.00009597, versus equal-CPU sampling standard errors0.00005982/0.00005705/0.00005979. These changes are not rigorous errors, but prevent a credible efficiency claim without a stronger reference. Independent262144-input truth estimates are0.47709656/0.50111771/0.36341858 with standard errors around0.00094–0.00098, too imprecise for that comparison.

The .0025 model used one16-node positive fallback at GL4 and one at subdividedGL8. Weighted fallback quadrature bounds were below4e-33; this excludes float geometry/roundoff and does not bound the dominant outer integration error. No moment guards failed after the fix. The shared T55 contribution is upstream86c3759; root integrated it separately.

### Cycle 5 protocol: independent cone reference
Reuse the T48 generic geometry implementation (read-only import, source bytes and SHA captured at import) and this folder's positive R4 volume primitive. No new cone engine. On the same original and two noisy models, enumerate all gate regions, clip exact winning/rank inequalities, and compare GL12/16/20 cone masses. Independently integrate the complete gate partition with GL16; check determinant sum16, same-orthant/L1 ray conditions, and one direct affine-map check per gate region. Record all hull/clip/omission counters and any moment/geometry failures. The analytic quadrature bound is conditional on correct floating geometry/arithmetic. No exact/solved label until independent geometry review. Alternative explanations: duplicate/missing regions can bias every quadrature order (partition/affine checks); model accuracy agreement can be accidental while ties are wrong (zero-logit and scaling checks follow after base cycle). CPU cap115 seconds, one thread, global lock.

### Cycle 5 result: stable cone reference
`python m16/approaches/small_h4_4/cone_reference.py /path/H4_T4_seed0.pth --geometry /path/small_h2_56/geometry.py` under the common lock/thread/memory settings reproduces `cone_results.json` with geometry SHA1e80b004aa6931de78b2e7e7bbb2212f37be3325a8c7ee176aaf2bce17b0194e (T48 da5cb18 source). All three models and three accuracy integration orders plus full-gate partition checks finished in20.06 CPU seconds.

| Noise | GL20 accuracy | Integration-only bound | Geometry + GL20 CPU |
|---|---:|---:|---:|
| 0 | 0.4770131259941187 | 2.705e-8 | 4.44s |
| .0025 | 0.5014929079254982 | 2.821e-8 | 4.52s |
| .02 | 0.3638126426226527 | 2.095e-8 | 3.62s |

GL16 and GL20 differ by at most3.4e-15; GL12-to-GL20 differences are below8.7e-12. Full-gate Gaussian partition sums differ from1 by at most2.3e-13, determinant sums equal16, affine-map checks differ by at most2.7e-14, and ray L1 deviations are at most4.5e-16. There were zero discarded degenerate pieces, zero determinant omissions, and zero whitening retries. Geometry costs above include full-gate triangulation and affine validation, making those CPU figures conservative for the estimator itself; extra accuracy orders and partition integrations are validation costs and remain reported separately.

[measured] The same code handles the original and two independent noisy copies without changing integration settings. [inferred] This is a strong small-case candidate: finite integration bounds are far below equal-CPU sampling uncertainty. [unverified] Floating geometry has not received independent T52 review, so neither total-error certification nor a solved label follows yet.

The independent cone reference reveals final angular biases of-3.04e-5/-1.25e-5/-7.87e-6. These are smaller than equal-CPU sampling standard errors for that particular refined angular rule, but the cone method is both more accurate and faster here. Numerical outer quadrature remains a distinct method class.

### Cycle 6 protocol
Package the fixed-order cone estimator with the standard tuple interface, explicit ignored-budget/CPU-cost limitations, and no random input use in its estimate. Verify zero/identical output-row tie ownership, Wi/Wo scale factors1e-16, and full-gate mass/determinant invariants on these controls. Reuse pinned generic geometry read-only; no changes to T48. No acceptance claim before root's independent review.

### Cycle 6 controls
`python m16/approaches/small_h4_4/cone_controls.py /path/H4_T4_seed0.pth --geometry /path/small_h2_56/geometry.py` reproduces `cone_controls.json`. Multiplying Wi or Wo by1e-16 changes the GL20 estimate by2.8e-16 or1.1e-16 respectively. All-zero and identical output rows both return0.24999999999999947, agreeing with first-index argmax accuracy1/4 to5.3e-16. These are author-run controls, not independent confirmation. `cone_estimator.est` now exposes the standard tuple API; its fixed GL20 ignores the requested budget and reports actual hardware-dependent CPU-calibrated forward equivalents. Timing-only zero-input evaluations are charged and never influence the estimated accuracy.

### Cycle 7 protocol: bounded per-cone order selection
For each same-orthant normalized simplex use Gram bounds a=max(1/4,min_ij Gij), b=max_i Gii for squared norm s. Expand1/s^2 on[a,b], with m=(a+b)/2, d=sqrt(ab), q=(sqrt(b)-sqrt(a))/(sqrt(b)+sqrt(a)). Chebyshev coefficients have absolute values2*q^k*(m/d^3+k/d^2) for k>=1. Sum the analytic tail after K=n-2 exactly, then apply the same positive-quadrature error argument. Choose the smallest order from2,3,4,6,8,12,16,20,24 whose error per determinant is<=1e-8/16. This allocates by geometry only and guarantees total integration error<=1e-8 for a disjoint partition with determinant sum<=16, conditional on ideal geometry/arithmetic. Alternatives: the tightened Gram bounds may be wrong (check known orthants and split cones against fixedGL20); adaptive selection may conceal a degree/off-by-one error (compare every model with independent fixedGL20 and record the summed actual bound). Test unchanged original/noisy models, save node/order counts and total CPU. No change to generic geometry.

### Cycle 7 result
`python m16/approaches/small_h4_4/adaptive_run.py /path/H4_T4_seed0.pth --geometry /path/small_h2_56/geometry.py` reproduces `adaptive_results.json`. Geometry-derived adaptive orders give accuracies0.4770131259941197/0.5014929079254998/0.3638126426226540, differing from fixedGL20 by at most1.7e-15. Summed integration-only bounds are6.88e-10/8.69e-10/5.98e-10, below the predeclared1e-8 target. Accuracy quadrature nodes decrease from80.856M/82.640M/61.856M to30.832M/26.687M/18.127M. Complete production CPU is3.15/3.00/2.29 seconds, including geometry. These times are measured separately from the earlier reference runs, which included extra geometry validation; no CPU speedup ratio is claimed from unlike workloads.

`cone_estimator.est` now selects these bounded adaptive orders with tolerance1e-8. It still ignores the advisory budget and charges actual CPU-calibrated forward equivalents. Order selection is per cone; the top-level order field is therefore null, with actual order counts recorded. A metadata-only correction changed the adaptive result's originally inherited fixed-order label20 to null; numerical results were unchanged.

Root independently reported original-model agreement to7.5e-14 using DRI double-description geometry and pulling triangulation, distinct from our Qhull engine, with the same positive volume formula. This supports geometry independently but does not independently validate the volume formula or certify every floating operation. Root/Claude own acceptance and registry status.
