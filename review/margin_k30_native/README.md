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
Source/provenance phase; independent source gate required before pilot.
Full continuation is not authorized by pilot code: review its completeness,
cost and resource projection first, then publish the concrete board extension.

## Results
Pending. Commands from repository root, one thread and shared lock:
`python review/margin_k30_native/provenance.py` and then independently gated
`python review/margin_k30_native/pilot.py`.

## Compute accounting
One thread, memory2GB, primary process105CPU soft/110CPU hard, explicit90CPU
pilot stop with saved partials. Provenance under review20CPU lane. Baseline and
instrumented evaluations are research work; two historical trained controls
reused once. Numeric111 sampling and historical72 remain separate. Integer,
control, memory, library internals and model-free setup excluded from numeric
source units, retained as separate categories where observed. No full runtime
or installed native instruction equivalence claim.

## Claims
[inferred] The reviewed tuple/source tracer can measure this unchanged recipe;
same-copy output and saved-value parity are mandatory, not assumed.

## Alternatives not ruled out
Alternative1: the two earlier copies have atypical operation counts; direct
fixed64 pilot and ultimately per-copy coverage discriminate it. Alternative2:
source drift/model reconstruction changes estimates; hashes and exact original
value/counter/cell checks discriminate before comparison. Native internals or
other omitted work may still change matched-cost conclusions. No full panel
classification follows from the pilot.

## Failure modes
Source/model mismatch, unsupported trace calls, output mismatch, CPU/memory
exhaustion or missing ledger cause an explicit incomplete result. Preserve all
failures; a finite failed trace does not retire the estimator family.
