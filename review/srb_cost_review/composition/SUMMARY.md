# Frozen SRB composition results

The sole fixed run4e5685df completed after source gatea8c1e552; raw publication49281e8c. Independent saved audit1f0aed36/shared08c33906 passes all four cases/eight arms, prototype and26 progress records in.100015065CPU without Q/estimator/RNG calls. No new candidate calls were made for this summary.

| Control | Budget | n | Kept M | Drawn D | Raw charge | Harness charge hex |
|---|---:|---:|---:|---:|---:|---|
| minimum quota |0|0|28|50000|48|0x1.8bffffffffffep+6|
| one batch |1000|588|584|50000|993|0x1.0001fffffffffp+11|
| forced second batch |1000|588|584|100000|993|0x1.0001fffffffffp+11|
| binary floor |17|9|28|50000|48|0x1.8bffffffffffep+6|

All four baseline/observed outputs and warnings agree. The eight toy calls use unchanged extracted source, the frozen model-free table, and deterministic API streams. They do not sample the Gaussian input law. All estimates happen to be zero for these toy controls; this is not an accuracy validation.

For the two budget1000 cases, the nominal draw allowance is292/5. Under the explicitly conditional policy interpretation that each kept prefix receives one free-input allowance, additional candidate charges are49416/25 and99416/25. The estimator discards D and reports the same993; the harness multiplies that entire integer by its binary3.3/1.6 factor. Replacing only the Q component would instead give the hypothetical1986, which is not an adopted or validated total cost.

The budget17 case preserves Python binary floor n=9. Minimum quotas force28 retained rows even at budget0. This describes the frozen interface; it is not a new allocation repair.

## Reproduction and accounting

Historical sole command (output paths refuse existing results):

```sh
ulimit -v 2097152
ulimit -t 20
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
flock -n /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python review/srb_cost_review/composition/run.py
```

Eight estimator calls and one seven-prototype stratum control;0.451354012CPU seconds and75336KiB maximum RSS. Zero Gaussian RNG/model-checkpoint/MC calls. Source extraction, table loads, deterministic stream construction, observation and serialization are research overhead, not claimed free algorithmic work.

## Scope and alternatives

No full native cost, historical candidate count, Gaussian frequency, estimator MSE or leaderboard classification follows. Different streams and budgets may have different rejection overhead. The old tie defect and cutoff remain; T210 repair is separate. T214 separately examines a conditional mandatory-cost floor across the entire frozen budget list, without reusing these toy streams as historical data.
