# Reciprocal T11597 saved control audit

## Question
Do all16 frozen sufficient-identifiability controls and their raw accounting
support the author's stated outcomes?

## Method
Claim4685f6ae preceded checker8d64b0c2. Independent determinant expansion of
all minors establishes rank and the row-space equality of saved RREF. Direct
subset sums independently reconstruct all fingerprints/collisions, and saved
progress/model/source pins plus source-shaped count contracts are checked.
No author imports, network calls or control reexecution.

## Status
Complete. The prior read-only source gate and T55 theorem review are separate.

## Results
All16 records pass:7 sufficient positives,5 inconclusive failed predicates,
4 malformed-input rejects;33 progress records/pins match. No failure occurred.
Run used `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python review/time_shift_reuse/fingerprint_review/check.py` with external `ulimit -v 2500000` and internal19/20CPU. Completed outputs cannot be overwritten.

## Compute accounting
.007975006 reviewer CPU; no author/trained/mass/model calls. Original reported
CPU0 is below author timer resolution, not a claim of zero computational work.
Exact arithmetic/validation/hash/IO remains research cost.

## Claims
[measured] All saved predicates and frozen ledger formulas agree independently.
[inferred] Successful sufficient predicates support the previously reviewed
real-arithmetic theorem only. Failed predicates do not disprove identifiability.

## Alternatives not ruled out
Domain-restricted equality, label-only equality, approximate merging and exact
reset states remain distinct. No trained model or cost-saving conclusion.

## Failure modes
No row-swap fixture occurs; observed reduced-bit maxima were not independently
replayed. Source arithmetic excludes decode/hash/GCD/control/storage work.
