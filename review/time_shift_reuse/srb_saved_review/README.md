# Independent saved SRB six-control audit

## Question
Do the frozen six Q fixtures support their saved output parity and named expression counts?

## Method
Published delegation25e98278; T55 yielded the saved audit. Checker parses original/observed AST without compiling, reconstructs original syntax, binds every event to an original operator/call site, verifies dynamic loop frequencies, broadcast/matmul and key call shapes, reconstructs category totals and source formulas. It checks fixed input/output byte encodings, warning parity, all progress events and pinned provenance. No author, NumPy, SciPy or Q calls.

## Status
Frozen before once-only saved audit.

## Results
Pending. Command: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash -c 'ulimit -v 2097152; ulimit -t 20; flock /workspace/work/codex-review.lock /opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3.12 review/time_shift_reuse/srb_saved_review/check.py'`.

## Compute accounting
Only parsing, hashing, byte encoding and integer ledger reconstruction; internal19/20CPU and external20CPU/2GiB. Review overhead separate from21 author paired evaluations/42Q calls.

## Claims
[inferred] Exact saved parity validates this observer on the fixed fixtures, not analytic correctness. The original zero-readout defect and native-cost exclusions remain.

## Alternatives not ruled out
Identical numerical bugs in original/observed paths remain possible; the zero-readout example is already such a counterexample. Unpriced native work and data-dependent interval behavior prevent multiplier transfer.

## Failure modes
No numerical replay, trained applicability, independent Gaussian correctness, full runtime cost or cost-ratio adoption. Call metadata contains shapes, not all values; original source gate supports observer provenance, this audit checks records/structural consistency only. Failures preserved without automatic replay.
