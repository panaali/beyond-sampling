# T240 independent saved synthetic audit

## Question
Do the fixed rational box/gap ownership, first-index margin and probability-bookkeeping records satisfy their stated contracts?

## Method
Explicit delegation `e053f1e1`. Checker frozen before outcomes at `af6bbd82`, with a numeric-argument type guard added at `43a7adf5`; source/resource review `dddae890` (shared `4abace2e`) cleared the sole saved run. Completed inputs and review gate were published at `27d9fb23` (shared `492fab40`) before execution. Author raw `bc358242`, shared `0a0ecb37`, remains unchanged.

`check.py` independently checks all Cartesian box vertices for whole-box ownership, scans critical second-coordinate endpoints for every ordered pair, validates saved witnesses and both feasible interval endpoints, and preserves boundary-only intersections. Margin statuses come from reconstructed logit winners at every supplied interval vertex with first-index tie ownership. Mass intervals use exact rational conservation and supplemental feasible endpoint triples. Guard reasons, original argument types, source/protocol/reviewer pins and all progress events are checked. No author implementation is imported or called.

## Status
Passed once; no failure, retry or author replay. `results.json` contains every independent row check and all completed-input pins.

## Results
All31 fixed cases pass: twelve boxes, six margin intervals, four mass-bound cases and nine guards. All64 progress events agree. Positive-width, nonnegative-gap, boundary-touch, first-index ties and zero-event cases remain distinct. These are synthetic supplied intervals, not trained network certificates or measured Gaussian event masses.

Exact command used once, with output overwrite prohibited:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash -c 'set -euo pipefail; set -o noclobber; flock /workspace/work/codex-review.lock bash -c "set -euo pipefail; ulimit -St 19; ulimit -Ht 20; ulimit -Sv 2097152; ulimit -Hv 2097152; timeout 25 /workspace/clean-neurons/.venv/bin/python review/t240_synthetic_saved_review/check.py" > review/t240_synthetic_saved_review/run1.log 2>&1'
```

## Compute accounting
Review CPU after standard-library imports:0.004007638 seconds; peak RSS25,564 KiB. External limits19soft/20hard CPU seconds,2GiB address space, one thread, review lock and25wall seconds after lock acquisition. The checker verifies its actual CPU/address limits and the recorded author's limits. Author0.083639321CPU is retained metadata, not independently timed. Zero author/CDF/RNN/geometry/Monte Carlo/RNG or trained-array calls; only exact rational saved arithmetic, hashes and text/JSON reading.

## Claims
- [measured] All fixed saved records, coverage/provenance and rational inequalities agree with this independent checker.
- [inferred] The supplied interval bounds support the corresponding conservative ownership/status/bookkeeping conclusions.
- [speculative] A later trained-domain certificate may use these prerequisites; no trained correctness, useful certified mass or cost advantage follows here.

## Alternatives not ruled out
Real gap-event coverage may be small, networks may remain difficult even on separated inputs, or stronger restricted-domain reasoning may help. The fixed toy inputs do not discriminate those possibilities. Any H8 or target-data application remains separately scoped and source/resource reviewed.

## Failure modes
Boundary-only feasible pairs are outer-cover candidates rather than positive-mass ownership. Supplied margin intervals are premises and not derived from a network here. Synthetic probability intervals do not certify actual Gaussian mass or finite CDF rounding. Guard auditing verifies saved error records and independent invalidity, not rerun exception behavior. No full software, numerical or cost completeness claim is made.
