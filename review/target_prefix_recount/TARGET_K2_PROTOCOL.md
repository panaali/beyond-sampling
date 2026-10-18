# Frozen target-byte / planar phase

Preparation is published under T219 board62878696. Execution requires the separate published source/resource gate AND completed independent saved18 review, both pinned in target_k2_gate.json. One attempt, no retry or precision/model expansion. This is only depth1/2, not k3/4. An author-count discrepancy is saved as a completed discrepancy, never suppressed by asserting the reported34. Disagreement between our independent planar and cone constructions is an implementation failure.

## Inputs and exact contract

Only the existing target checkpoint498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb and existing T201 weights archivefea77e7cb6e6e23ba2dcba71466b1cd69ac31063f76a3a949fb5f7ea8ab63466 are decoded. Pin the T201 pilot source, metadata and independent saved review, plus the architecture/loading sources used for the target contract. CPU-only weights_only=True checkpoint load once; no model construction, forward, new data, fitting, RNG or mass. Verify exact keys,432 original float32 bytes and all converted float64 bytes. Wo160 is provenance only; common exact power-of-two denominator uses Wi16/Wh256. Save all272 coefficient ratios, integers and D with exact reconstruction checks. The counted model is the bias-free zero-initial-state real-arithmetic recurrence on these dyadic values, not a claim about every floating execution of the author code.

## Independent constructions

First compute the planar partition by exact scalar Fraction roots as proved in TARGET_PLAN.md. For each x0 sign, list all unique roots, affine rows, interval representatives, masks, boundary rays and hidden maps. Tail rays have x0=0 and x1=+-1; all finite boundaries use positive denominator directions. Zero input coefficients contribute no roots, constant/zero forms keep their exact sign/OFF convention. There are no input samples and no calls from this planar routine to core.py.

Then invoke the reviewed core.py only for depth1 and2. Save every cone/history/hidden matrix and exact strict witness certificate. Compare every resulting history, ray set and hidden map with the independent planar partition. Save the author34 comparison separately, preserving either result. The analytic controls and the theorem supply the complete-ray induction; finite witnesses alone are insufficient.

## Resources and durability

Exactly one20CPU/2GiB reviewlock invocation,16CPU internal soft cap, one thread for OMP/BLAS/MKL/Torch. NumPy and Torch imports, decoding, exact preparations/constructions, validations and serialization are charged as research CPU. Save per-stage attempts before work and atomic results between stages; existing result/temp/progress prevents rerun. Preflight errors are preserved in the launcher log. Resource or arithmetic failure preserves partial dictionaries and remains inconclusive. Named exact cone counts exclude uninstrumented Fraction/preparation/comparisons/gcd internals and are not native or scalar-FLOP prices. Peak retained bits omit some intermediate values. No inference about a competitive estimator, total T10 regions or noisy-model transfer follows.

Command only after both gates are published/pinned:

    flock /workspace/work/codex-review.lock bash -c 'ulimit -t 20; ulimit -v 2097152; export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; /workspace/clean-neurons/.venv/bin/python review/target_prefix_recount/target_k2.py'
