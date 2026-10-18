# T261 tiled saved metric audit

## Question
Does the tiled candidate's retained fixed-eight panel satisfy the same summary/coverage contract?

## Method
Byte-identical independent check.py from the original saved audit, new scope/pins and outputs. One20CPU2GiB review-lock read of saved metrics; no candidate/tensor/author replay. recover_910 audits attempts, progress and provenance separately.

## Status
Passed all eight identities and112 metric records, including11 exact-zero-reference records and all recorded equality flags. Completion is summary-level, not independent reconstruction from tensors.

## Results
Maximum absolute error2.384185791015625e-7, relativeL2 error3.932716176130984e-7. Exact retained result contains authoritative final digits/CPU. Both remain below frozen2e-6/1e-6 limits; no tolerance change. Original whole-batch failures and its separate successful parity remain preserved.

## Compute accounting
One saved-only audit; actual20/20CPU and2GiB. No model, gradient, optimizer, compiler or RNG execution. Reproduction command is the original saved_metrics/check.py command with tiled_saved_metrics path; completed outputs refuse overwrite.

## Claims
[measured] Saved summary and coverage consistency pass. [unverified] Independent tensor-level or CUDA parity, graph/checkpoint runtime and throughput. Compiler metadata/PTX binding is separately reported in ../tiled_compile_saved_review.json.

## Alternatives not ruled out
Shared memory now fitting the nominal per-block budget does not exclude register spills, low occupancy, launch incompatibility or numerical GPU differences. Other tiled/reduction designs remain available.

## Failure modes
Only author-produced metric summaries/equality flags are retained, and the local-update sequence does not bound independent trajectory drift. No adoption or speed claim follows.
