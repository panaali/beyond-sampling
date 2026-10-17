# T186: direct missing-k30 RECIPE85B cost coverage

## Question
Do direct costs preserve the existing fixed training-panel comparison? Two prior
copies cannot supply a cost multiplier for the remaining 4,094 copies.

## Method
Claim9235ef05. Reconstruct all existing 4,096 seed1..4/noise20..23 models from
their saved checkpoints and exact RNG order. `provenance.py` verifies every hash
without estimator calls. `pilot.py` freezes first64 seed1/noise20 copies and
reuses the reviewed T176 copy0 once, retaining unchanged source, original output,
legacy cost, cells, STATS and grouped errors. Complete ledgers are losslessly
compressed. No new truth, training, noisy weights or recipe choice.

## Status
Complete bounded missing-k30 panel. Independent pilot/source gate7e877343
preceded continuation b71c9744/ef1d71c8. All4096 unique hashes reconstructed in
.067944CPU without estimator/truth calls. Full independent saved audit passes;
final review hash is recorded in HANDOFF.md. No shared estimator/INDEX edits.

## Results
Full reconciliation2a7a64a9 preserves every original value/counter/cell and
baseline/traced STATS plus grouped errors. Direct numeric434,926,019 and
integer13,500,632. Numeric111 own-integer-budget comparisons:

| Training seed | Ratio | Conditional bootstrap95 | Individual losses | Worst individual |
|---|---:|---|---:|---:|
| 1 | .06470896873 | [.05578395,.07527001] | 3 | 1.70745155 |
| 2 | .00014096110 | [.00011513,.00017119] | 0 | .00763702 |
| 3 | .01345273342 | [.01185315,.01537496] | 0 | .60100733 |
| 4 | .24464468580 | [.20335005,.29149920] | 45 | 12.45500052 |

All16 noise-block averages are below1, worst.26346366. The48 individual
losses remain, so this is not every-copy success. Historical72 has25 losses;
seed4 matched integer ratio.15943546. Its earlier.16013643 is the mean-budget
comparator, not a changed estimate. Original mean/continuous comparisons
reconstruct in historical_reconciliation.json. Only seeds1..4 noisy copies and
RECIPE85B are covered: original seed0, clean and RECIPE112 direct costs remain
outside this claim. No full training-rung statement follows.

Pilot c7f45934/73ed7c21 retains numeric6,327,190 + integer202,274 and ratio
.0795080 versus historical.0527719, no individual pilot loss. Pilot10.569564315CPU
and206072KiB gave rough676.452CPU projection; full observed work is below.

Commands from repository root, one thread and shared lock:
`python review/margin_k30_native/provenance.py`, gated
`python review/margin_k30_native/pilot.py`, then
`python review/margin_k30_native/shard.py SEED NOISE BLOCK` with seeds1..4,
noise20..23, blocks0..3 except reused pilot(1,20,0). Saved-only contraction:
`python review/margin_k30_native/full_summary.py` and
`python review/margin_k30_native/reconcile_historical.py`. Completed artifacts
refuse overwrite; independent reproduction uses a separate owned copy.

## Compute accounting
One thread,2GB,90CPU graceful stop,105CPU soft/110CPU hard. All64 shards complete,
current research CPU745.682597126; max13.303206964CPU, peak206256KiB.
Historical two-control CPU2.368110423 is separate and was not rerun. There were
8188 new estimator evaluations (plain and traced for4094 copies), no reference
or model-input Monte Carlo/truth evaluations. Summary1.871870557CPU and
provenance.067943864CPU. Timers include tracing/parity/I/O, not production
estimator speed; bookkeeping/import timing limitations remain. Pilot projection
was an estimate, not a total-runtime cap.

Numeric111 sampling and historical72 remain separate. Integer/control/memory,
library internals and model-free setup are excluded from numeric source units,
retained separately where observed. No complete native/hardware/runtime cost.

## Claims
[measured] Every claimed copy has its own source ledger and unchanged original
estimate; no cost multiplier transferred.
[measured] All four seed averages are favorable under declared numeric111 and
saved truths;48 individual copies lose. Bootstrap is pointwise conditional,
not a simultaneous population/geometry/reference certificate.
[inferred] Full runtime conclusions require omitted-category and matched
sampling accounting beyond this source ledger.

## Alternatives not ruled out
Alternative1: two earlier copies have atypical costs; complete direct per-copy
coverage now measures this possibility rather than assuming a multiplier.
Alternative2: source/model drift changes estimates; every hash and original
value/counter/cell plus grouped-error parity passed. Native internals or omitted
work may change matched-cost conclusions. Seed0/clean/112 costs and other
training levels remain separate follow-ups. The48 individual losses show the
aggregate result does not establish a uniform winner; no family retirement.

## Failure modes
Source/model mismatch, unsupported trace calls, output mismatch, CPU/memory
exhaustion or missing ledger cause an explicit incomplete result. Preserve all
failures; a finite failed trace does not retire the estimator family.
