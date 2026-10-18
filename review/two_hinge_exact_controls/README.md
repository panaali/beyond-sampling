# T233 exact residual controls

## Question
Do the reviewed pointwise residual, conditional interval and gauge identities
survive an exact implementation, including anchor-zero boundary ownership?

## Method
`primitive.py` uses Fraction arithmetic only. `fixtures.json` fixes twelve
five-point cases; `controls.py` checks original ReLU differences, endpoint
polynomial integration and gauge invariance, plus eight malformed/domain guards.
`PROTOCOL.md` limits the one possible synthetic attempt; `protocol.json` binds
sources and original proofs. No target data or Gaussian evaluation is involved.

## Status
Sole fixed synthetic attempt complete in d5721b03 after source review 6a57dfe7
and published execution gate a21d5ce2. Independent saved-output audit is active
with recover_fundamentals; no further run is authorized.

## Results
[measured, saved audit pending] All twelve five-point cases and eight guards
passed; 42 durable events. Interface counts: 120 point, 60 interval and 12 scale
calls, plus eight rejected guard attempts. Main-scope CPU: 0.427782553 seconds.
The runtime records CPU soft/hard [20,20] seconds and address-space soft/hard
[2147483648,2147483648] bytes. Reproduction command (original run only; outputs
refuse overwrite): `set -e; ulimit -v 2097152; ulimit -t 20;
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 flock
/workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python
review/two_hinge_exact_controls/controls.py`. The irrational density maximum is
kept symbolic, not evaluated or counted as a computed Gaussian expectation.

## Compute accounting
Counts and measured main-scope CPU above include the synthetic phase. There
were zero model/MC/RNG/quadrature/Gaussian calls. The 84 successful nested
conditional calls are distinct from interface counts. Exact arithmetic, GCD,
validation, snapshots and imports are not given a scalar/native cost model.
External limits cover the whole process; no performance benefit claimed.

## Claims
[inferred] Exact rational arithmetic avoids rounding in the tested identities.
[speculative] These identities may support a later useful conditional residual
method. No target rarity, cheap integration, accuracy or scaling claim follows.

## Alternatives not ruled out
Stable Gaussian expectation, non-Gaussian/truncated conditional laws, multiple
anchors, alternate orientation and source-cost-aware aggregation are untested.
The signed surrogate can be negative and must not be silently used as a valid
hidden state without restoring/bounding its residual.

## Failure modes
Endpoint ownership, slope-domain confusion, using area as expectation, dropping
mean terms in later state moments, and mistaking small expected margin changes
for small classification error. Any failures stay in immutable records.
