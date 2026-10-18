# T261 independent saved metric and coverage audit

## Question
Do the retained metric summaries cover every frozen interpreter comparison and satisfy their declared tolerances?

## Method
A separate stdlib/Fraction checker binds saved validation bytes, checks the eight exact case identities, fourteen records per case, explicit zero-reference handling and recorded equality flags. No author, Torch, Triton, tensor or RNG calls. Command: `flock /workspace/work/codex-review.lock bash -c 'ulimit -St 20; ulimit -Ht 20; ulimit -Sv 2097152; ulimit -Hv 2097152; timeout 25 /usr/bin/python3 review/fused_rnn_source_review/saved_metrics/check.py'`. Completed output refuses overwrite; do not rerun it.

## Status
Passed saved-summary consistency; recover_910 independently audits attempts/resources/progress/source binding. Actual raw tensors are unavailable, so this is not an independent reconstruction of numerical errors.

## Results
All eight case rows and112 metric records pass;11 zero-reference records have exact zero recorded error. Maximum absolute error2.384185791015625e-7 and relative L2 error2.8502714512413043e-7 are below2e-6 and1e-6. All recorded accuracy/step/inactive equality flags are true. These flags remain harness evidence. The three sequential local controls do not establish independently evolving trajectory agreement.

## Compute accounting
One saved-only checker, .002779163 CPU seconds, actual soft/hard20CPU and2GiB. No candidate/reference/compiler/model calls or numerical tensor replay. Prior owner failed attempts remain separately charged.

## Claims
[measured] Retained summary coverage and threshold consistency pass. [unverified] Full tensor parity independent of the author metric code. [unverified] CUDA numerical behavior, adapter graph lifecycle, hardware launch feasibility and throughput.

## Alternatives not ruled out
A CUDA-only failure or large-batch resource failure can remain despite interpreter success. Tiling and different pipeline staging remain distinct variants requiring scoped checks.

## Failure modes
Rounded aggregate metrics cannot recreate tensor values or full original norm-product predicates. The offline compiler's successful code generation is not evidence its shared-memory request fits hardware; resource metadata must be checked separately.
