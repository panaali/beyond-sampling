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
