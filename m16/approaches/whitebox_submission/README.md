# T479: first White-Box submission

## Question

Can the frozen T474 memory-safe public baseline complete an official Phase 2 submission?

## Method

Ship T474's `estimator_blocked.py` byte-for-byte as `package/estimator.py`, with the
upstream MIT license. This derives from 504aldo V25, not an original cumulant
algorithm. `whest package` supplies the manifest. No research logs, references,
credentials or dataset files belong in the upload.

## Status

Eight submission uploads completed before the cutoff. Official IDs and archive hashes are recorded in `submissions.json`; all were queued awaiting holdout compute at the final browser check. Grading and private nomination remain external follow-ups. No organizer messages authorized.

## Validation protocol

Before upload, validate the exact packaged source and rehearse one official-shape
prediction with the official subprocess runner, one BLAS thread, 120-second
prediction cap and 0.4-second residual cap. Use only 128 reference samples for
this interface smoke test; its score is not an accuracy estimate. Prior accuracy
measurements remain in INDEX T474. Initial guard: every owned process below 1.65 GB RSS; owner subsequently waived the local memory cap. The rehearsal now uses an 8 GB guard aligned with the remote process limit,
limit the rehearsal to 240 seconds wall time, and record failures without silently
changing the estimator. Inspect archive members and hashes before upload.

## Results

See INDEX T479 when registered; validation artifacts live in `results/`.

## Compute accounting

Subprocess smoke tests for the three base families, including retained failed attempts; prediction uses the official FLOP meter. Exact-source small-shape validation for all eight archives. Variant full-shape numerical checks and independent source reviews are recorded under T475/T476.
The small reference and package/validation work are preparation costs, not a
claimed score improvement. No additional model fitting or parameter search.

## Claims

- [measured previously] T474's local evidence is linked in INDEX, not duplicated.
- [speculative] The same source passes the remote grader; only an official
  submission result can establish that.

## Alternatives not ruled out

Remote residual-time overhead may exceed local measurements. A local passing
interface or isolated process is not the remote sandbox. Stronger public methods
and error attribution remain separately queued as T475–T478.

## Failure modes

Residual-time cap, numerical failure on unseen models, import/sandbox differences,
submission deadline, account eligibility or platform failure. No official rank
is inferred from local test results.

### Rehearsal interpretation

The initial low-memory subprocess attempt stopped in the harness. After the owner
waived that cap, V25 and e30 passed the official subprocess path. Logxio exceeded
the residual cap during concurrent experiments; a separately retained isolated
repeat passed. Both receipts are retained. Its timing margin is therefore less
robust than e30. Tiny-reference smoke-test MSE is intentionally not used for
accuracy or rank claims.

### Submission coverage

The eight archives cover blocked V25 (128/256 rows), original e30, original logxio, e30 RPC dtype compatibility, e30 rank448, pruning threshold3.0, and ranks384/192. They retain upstream MIT licenses. Full subprocess rehearsals cover base families; variant checks do not independently establish remote RPC timing. Two unused slots were left rather than uploading known worse or timeout-prone experiments. The selection page states that the two best public scores enter private re-evaluation automatically if no manual selection is made; selection closes 2026-10-22 23:59 UTC. Screenshot proof: `results/submissions.jpg`.

### T482: final owner-requested attempt

Owner requested one more entry at the cutoff. Combined the already reviewed rank448 candidate with the previously reviewed RPC dtype comparison fix, with no additional fitting or numerical changes. Exact-source tiny validation and official packaging passed; `results/rank448_rpc.diff` verifies the single change relative to the six-network tested rank448 source. No full-shape run of this combination was completed. The website closed submissions at 23:59 UTC and rejected the upload-page navigation, so this archive was **not submitted** and has no official ID. See `results/last_attempt.json`; accepted entry count remains eight. More local compute would not reopen the deadline.
