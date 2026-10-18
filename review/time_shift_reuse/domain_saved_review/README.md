# Independent T11596 saved64 audit

## Question
Do the fixed64 saved distinct-state comparisons reproduce from the pinned raw census and exact weights?

## Method
Delegated TASKS publication12c8b9bb; T55 yielded this saved review. `check.py` imports no author code. It reconstructs exact state representatives, uses triangular-block pair-index inversion, binds raw rays/maps/models, computes WhM, signs, canonical and projective keys, and closed-form named operation counts. It checks all134 progress records. Peak-bit internals are excluded.

## Status
Frozen checker before once-only execution; author result cd04347e is unchanged.

## Results
Pending. Command: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash -c 'ulimit -v 2097152; ulimit -t 20; flock /workspace/work/codex-review.lock /opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3.12 review/time_shift_reuse/domain_saved_review/check.py'`.

## Compute accounting
Only JSON reads, SHA256 and exact rational saved-record reconstruction. Internal19/20CPU, external20CPU/2GiB. Zero author calls, geometry, forward, random, reference or mass evaluations.

## Claims
[inferred] Matching records support this sufficient-method execution only. All noncertificates remain unknown.

## Alternatives not ruled out
Actual recipe-domain feasibility and lower-dimensional completeness require separate evidence. Audit distinguishes a wrong selector/raw binding from a correct finite result; neither a passing audit nor128 unknown hulls establishes impossibility.

## Failure modes
Peak-bit internals, actual runtime cost, label equality, probability and whole-tree coverage are not checked. Durable failure records will be retained; no automatic reruns.
