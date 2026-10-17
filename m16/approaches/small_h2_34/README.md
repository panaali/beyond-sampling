# T47: two-neuron T3 and T4 recovery cycles

## Question
Can weight-derived calculations solve the accuracy-estimation problem at H2,T3 and H2,T4 at competitive compute?

## Method
Recovery protocol frozen before new runs. The original unpushed implementation was lost during the environment restart. Its numbers survive only in conversation and are not treated as present artifacts.

Cycle 1: reuse T55's analytic two-input tail. At T3 the normalized one-input prefix has only two directions, -1 and +1, equally likely. Thus two tail probabilities averaged give full accuracy analytically, with floating-point polygon geometry. Compare to independent `facet_transport` three-step octant-first enumeration. Use recovered public seed0 arrays plus eight absolute-sigma0.05 noisy copies, RNG47001, matching the original protocol. Check zero/duplicate logits and positive scaling. Do not repeat million-input MC validation: independent geometry provides a more precise reference. Benchmark elapsed process CPU against a bounded 65536-input direct forward batch; report hardware-dependent CPU equivalents separately from primitive counters. No random model inputs in the estimator itself.

Two alternatives before runs: (1) agreement could reflect shared geometry bugs; use different partition order and analytic constant-output cases. (2) apparent computational advantage could depend on incomplete counters or vectorization; report entire CPU cost including geometry and separately show matrix/clip/moment counts, with no operation-based solved claim. (3) the trained checkpoint could be atypical; include all eight frozen noisy copies, not a selected winner.

Cycle 2 planned: T4 normalized prefix is a uniform circle; integrate the T55 conditional probability over its angle with the H4T4 owner's reusable driver. Adaptive integration is numerical deduction over weight-derived probabilities, not an exact closed form. Require refinement, independent checks, and honest full cost.

## Status
Running; root owns independent review and Claude is DRI. Existing T47 claim remains active. Each cycle <=120 CPU seconds, numerical work serialized with flock, one BLAS thread and <=2GB per-process memory. No shared files edited or remote pushes by this worker.

## Results
Pending recovered artifacts. Old T3 report was accuracy40.9012--41.1308%,38--39 rank/gate cells, but will be reproduced independently. Old T4 inverse-covariance quadrature had warnings and shifted by3.96e-5; it is not accepted as a reference.

## Compute accounting
Primitive counts and full process CPU are reported separately. CPU-equivalent forward counts include implementation overhead but depend on hardware and batch size. No FLOP advantage follows merely from elapsed time. Sampling baseline has exact Bernoulli variance p(1-p)/N when p is supplied by independently checked geometry.

## Claims
[inferred] T3 reduces to two analytic conditional integrals. [speculative] Their cost may be competitive. No self-confirmed or solved claim.

## Alternatives not ruled out
General noise ensembles and other training seeds may behave differently. Full challenge M16,T10 remains separate. T4 outer quadrature may have difficult angular breakpoints.

## Failure modes
Positive-volume output ties; floating thin polygons; cancellation of moments; incomplete operation counts; underestimating error from two agreeing numerical implementations. Numerical geometry is not a real-arithmetic certificate.

### Cycle 1 cost addendum, frozen before accounting revision
The recovered two-tail estimator agrees with independent geometry on all nine copies (maximum1.67e-16); new CPU benchmarks are saved. Before changing its interface cost, use an intentionally loose scalar-operation bound: for T3 each polygon has at most V=3+2H+2+2=2H+7 vertices (initial triangle plus gate, readout, and rank planes). Charge1000(V+3) scalar operations per clip and per moment call,100(H+1)^2 per affine product,1000(H+1) per final gate cell, and1000(H+1)^2 prefix/setup overhead. This covers arithmetic, comparisons, norm/division, polygon area determinant and trig calls as unit operations with a large margin; it excludes memory allocation/Python dispatch, which full CPU timings include. It is a proposed conservative arithmetic bound for review, not a hardware-FLOP count. Use the same conventional forward arithmetic proxy2[T H(H+1)+TH] to express this bound in FPE; no calibration model forwards belong inside est(). Verify the copied checkpoint arrays against restored files before finalizing the cycle.

### Recovered cycle 1 result
`results_t3.json` reproduces the full nine-copy accuracy range0.40901220810786887--0.4113077285977156 using T55's two-tail estimator. The independent octant-first facet calculation differs by at most1.67e-16. Both constant-output controls give1/3; both independent geometry partitions have mass one within floating precision. Positive Wi/Wo rescaling checks pass. The restored checkpoint SHA256 hashes and every stored weight were exactly verified.

The complete first run used0.529 CPU seconds, including nine independent geometry calculations, nine rescaled estimators, two tie controls, and27 timed batches of65536 direct forward inputs (1,769,472 benchmark inputs). These benchmark outputs are not used as probability estimates. Per-estimator CPU time corresponds to248,264--359,077 batched direct forwards; equal-CPU sampling RMSE is about0.08--0.10 percentage points. This is a CPU comparison, not a FLOP claim. Proposed conservative arithmetic bounds are stored separately and await root's independent audit.

Reproduce from repository root:
```
flock /workspace/work/codex-compute.lock bash -c 'ulimit -v 2500000; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/clean-neurons/.venv/bin/python m16/approaches/small_h2_34/run_t3.py'
```
The shared arc_eval is hardcoded to T10 and imports large calibration arrays, so this bounded ladder driver retains its model-tuple/est interface without importing that incompatible harness. DRI must adapt shared evaluation before claiming shared-leaderboard acceptance.

### Cycle 2 frozen recovery protocol
Use H4 owner's generic circle driver unchanged on the verified H2,T4 base checkpoint. First GL2/4/8 per octant, then GL16/32 if no numerical guards fire and the120 CPU-second cycle cap permits. Save every level immediately. One65536-input Gaussian validation batch with RNG47004 is a coarse check only. Noise-copy sweep is deferred until base angular convergence and moment stability are adequate; this revises the pre-outage nine-copy T4 plan explicitly. Alternatives: narrow angular features may evade GL (test successive orders and offset); positive-moment guards may reject valid nearzero regions (report failure, request owner fix, never clip/skip). No solved claim based on noisy MC agreement.

### Cycle 2 result
H2T4 seed0 angular GL orders2/4/8/16/32 gave0.3089245331/0.3076513113/0.3076442083/0.3077707293/0.3077559052. All conditional partition guards passed. Convergence is nonmonotone: the GL8-to16 shift1.27e-4 is material, so neither the low-order value nor the old Plackett output is an adequate reference. Cycle results are in `results_t4.json`; cumulative compute includes the65536-input coarse validation and five rules. Two initial runner attempts failed while assembling output after GL2 (duplicate dictionary key), before any results were saved; roughly one additional GL2 computation each was spent and is not in the final file's CPU total.

Reproduce using `run_t4.py --engine m16/approaches/small_h4_4/estimator.py` under the same flock/thread/memory wrapper. The T51 engine must be integrated alongside this folder. Next bounded cycle will use GL64/128 and an offset rule to test the apparent stabilization, without further Monte Carlo.

### Cycle 3 refinement protocol
Before further measurements: preserve old octant GL64 as `refinement_t4.json`; then use T51 owner's analytically located prefix-gate boundaries plus the original octant/rank edges. Test GL4/8/16 and midpoint subdivision2 at GL8. Prefix-gate partitioning is a different deterministic rule, so compare both sequences rather than treating one favorable estimate as truth. No additional MC. Fixed labels/tie constraints remain in T55. Every completed rule saved and total CPU per invocation capped110 seconds.

### Cycle 3 result
Old octant GL64 returned0.3077259838 (5.06 CPU seconds). Exact prefix-gate partition GL4/8/16 returned0.3077323820/0.3077594030/0.3077378631 (0.70/1.03/1.98 CPU seconds); splitting each prefix interval in half with GL8 gave0.3077213391 (2.01 CPU seconds). Thus exact prefix boundaries improve the starting partition but do not remove all angular features; the remaining3.8e-5 spread prevents using one low-order rule as truth. Files `refinement_t4.json`, `prefix_refinement_t4.json`, and `prefix_split_t4.json` preserve all nodes and counts. Total additional numerical compute about10.8 CPU seconds. No new MC. Continue toward an independent full-cone reference and additional convergence; do not claim certified bias from these differences.

### Cycle 4 independent reference protocol
Before running: reuse T48's dimension-generic cone enumerator and T51's stable positive simplex integral, both already owned by other workers. On H2T4 seed0 only, enumerate all gate cells and correct pieces in R4, triangulate, and evaluate correct mass at GL12/16/20. Report the analytic integration-only error bound and all geometry counters/omitted-mass diagnostics. Validate full gate triangulation determinant sum16 (signed-orthant cross sections), direct hidden affine maps at gate centroids, and gate probability sum at GL16. These geometry/rounding checks do not convert the floating polytope representation into a certificate. Alternatives: an angular discrepancy may come from slow circle convergence (compare independent cone reference), or from faulty cone clipping/tessellation (check full mass and affine maps before using it). One model, no random validation, <=110 CPU seconds and shared lock/memory cap.

### Cycle 4 result: independent H2T4 cone reference
`cone_reference_t4.json` supplies a materially stronger reference:117 gate cells,99 correct pieces and569 correct simplices. Positive mass GL12/16/20 gives0.30773074428588015/0.3077307430225551/0.30773074302392744. The GL20 analytic integration-only bound is1.78615e-8; GL16 bound1.16789e-6. Total cycle cost0.478 CPU seconds, including full gate triangulation/mass and all three integration orders. No random/model-input samples were used.

Checks: gate determinant sum15.999999999999996; gate probability0.9999999999955301; centroid hidden-map error1.12e-14; ray L1 error2.23e-16; all rays share their simplex's signed orthant; no degenerate pieces omitted. Correct and gate simplex quadratures total9,729,536 nodes, with only four-dimensional polynomial/rational arithmetic at each node. These are geometric integration nodes, not network input evaluations.

This establishes a precise numerical reference conditional on floating geometry, not a full formal certificate. It resolves the angular-rule discrepancy: prefix GL4 happens to be close, while increasing its order is not monotonically beneficial. Next cycle tests the direct cone method on all eight predeclared noisy copies, with fixed GL16 estimator and GL20 reference; geometry and all computation charged.

### Cycle 5 frozen ensemble protocol
Use the same H2T4 base plus eight absolute-sigma0.05 independent copies from RNG47001, preserving the pre-outage protocol. Fixed estimator GL16 over exact-formula cone simplices; GL20 is a numerical integration reference using the same geometry. It is not an independent geometry reference. Record each model separately, analytic integration-only bounds, geometry omissions, ray invariants, full gate determinant sum, all CPU and quadrature nodes. Equal-CPU sampling baseline uses one shared65536-input timed batch per model, including input generation, network forward, target sorting, and predicted argmax. Its output is a coarse validation only. This adds589824 samples and avoids expensive MC truth. Counterexample alternatives: a cheap exact-looking answer could miss/duplicate geometry (gate determinant and conditional angular comparison on base); underestimated cost could omit triangulation/rule generation (whole estimator CPU and primitive counters). Root independently validates further cases before any solved claim.

### Cycle 5 result: nine-copy direct cone benchmark
All nine predeclared H2T4 cases completed in3.463 CPU seconds total, including GL20 references and589824 coarse validation inputs. Accuracy spans0.3053576942--0.3098490642. Fixed GL16 differs from GL20 by at most1.43e-12; its analytic integration-only error bounds range1.15435e-6--1.17893e-6. Every gate determinant sum equals16 within3.56e-15; no geometry pieces were omitted. Coarse MC checks are within1.74 standard errors.

Entire estimator CPU per model is0.207--0.401 seconds, equivalent to986048--2147930 complete sampled accuracy evaluations on this machine. Even the conservative GL16 integration-bound squared, divided by equal-CPU sampling variance, is at most1.38e-5. That comparison is conditional on correct floating geometry: it is not a formal end-to-end error bound, nor a scalar-FLOP certification. GL20 shares the same geometry, so their agreement independently checks integration refinement, not clipping/tessellation. Root's independent geometry/tie/noise review remains necessary before acceptance.

`cone_estimator.py` supplies a fixed-rule T4 `est(model,budget,rng)` interface; budget is advisory and returned work is explicitly CPU-equivalent forwards, with deterministic timing calibration charged. It reuses T48 geometry and T51 stable integration without duplicating their engines. The older T3 `estimator.py` reports its separate conservative arithmetic cost class. Reproduce the ensemble with `run_cone_ensemble.py --geometry m16/approaches/small_h2_56/geometry.py --mass m16/approaches/small_h4_4/stable_volume.py` under the shared lock/memory/thread wrapper. Geometry/mass source hashes are recorded per row.
