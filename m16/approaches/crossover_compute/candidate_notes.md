# Candidate: conditional Gaussian delta method

Frozen design before test-model evaluation: enumerate all ordered positions of the maximum
and second maximum. Partition the second-largest value into equal-probability strata, compute
the exact conditional first two input moments by model-free one-dimensional integration,
linearize the weighted recurrence at each conditional mean, and integrate Gaussian marginal
failure tails. Sum marginal failures (first-order rare-event union approximation), clipped to
[0,1]. No fitted parameters, random inputs, or test-model tuning.

Competing explanations and cheap checks, specified before evaluation:

- Frozen gates and Gaussian input closure may erase the rare broad error bands. Check the
  signed bias at each predefined refinement level and across all available model groups.
- The first-order union may double count overlapping failures. Record the gap between the
  largest marginal failure and the sum. These bound a union of the approximate Gaussian
  marginals, not the true network failure probability.

A losing result will be retained without searching for test-set-specific corrections.
The minimum exhaustive chamber calculation may exceed a requested budget; reject that
budget explicitly instead of returning a favorable constant estimate.

## Protocol and accounting

The estimator was frozen in commit `895bef8` before any test-model evaluation.
`estimator.py` provides the shared `est(model, budget, rng)` interface and ignores `rng`.
`candidate_experiment.py` uses `leaderboard.models()` and the existing 28M/4M truth
cache, without writing shared `results/`. Results are provisional diagnostics: the
DRI has not approved the estimator's class or FLOP ledger.

There are 90 ordered top-two position chambers per second-value stratum. The frozen
levels use 1, 2, or 4 strata, costing 587.544, 1174.944, and 2349.744 FPE respectively.
Budgets 100, 250, and 500 are explicitly rejected. Budgets 1000, 2000, and 4000 select
the three respective levels. The ledger charges recurrence, full input Jacobian
propagation, logit Jacobians, covariance products, Gaussian tails (200 FLOPs each),
scalar operations and weight scans. Exact code is `cost_for`. Python dispatch and
memory traffic are not FLOPs: the reported CPU ratio diagnoses whether the abstract
ledger predicts actual efficiency. Model-free moment integration has a separately
reported conservative allowance of 10M FLOPs; it is cached across models.

MSE subtracts each cached truth variance. The baseline is the **mean of per-model
`p_i(1-p_i)/actual_cost_i`**, rather than v3's `mean(p)*(1-mean(p))/mean(cost)`.
The former is the expected sampling MSE on the actual finite model set. Therefore
this table is a compatible diagnostic input for the crossover report, not a DRI
leaderboard output. Group CIs bootstrap paired model error and baseline contributions
(5000 replicates); they are provisional with only 5 or 6 models and omit finite-truth
uncertainty. The single-model original group has `ci: null`, because no model-bootstrap
uncertainty can be estimated from one deterministic value.

Sanity checks reconstruct unconditional first and second normal-input moments from
the conditional moments, check positive covariance eigenvalues, test the exact
1/10 accuracy of a zero-logit network including argmax ties, verify rejected budgets,
and repeat every evaluated estimate with an RNG to check determinism.

The union gap is the difference between the clipped sum of Gaussian marginal failure
tails and the largest tail. It only bounds overlap uncertainty within the approximate
marginals; it is not an error bound for the nonlinear network. Gaussian closure assigns
mass outside the rank chamber; gate crossings are ignored; more strata do not prove
convergence to the true network probability.

## Measured result: negative

No observed winning tested point. This does not estimate a crossover below the
minimum supported cost, and does not rule out other mechanistic families.

| Actual cost (FPE) | orig | seeds | sigma .0025 | sigma .005 | sigma .01 | sigma .02 |
|---:|---:|---:|---:|---:|---:|---:|
| 587.544 | 201.278 | 1366.799 | 129.669 | 59.390 | 17.884 | 14.523 |
| 1174.944 | 87.117 | 2454.608 | 41.586 | 14.824 | 3.120 | 9.783 |
| 2349.744 | 319.129 | 4706.926 | 153.912 | 81.509 | 22.656 | 22.037 |

Entries are MSE / equal-cost sampling MSE; lower than one would win. All group
mean biases are negative. Original biases are -12.368, -5.754, and -7.788 percentage
points. Seed-group biases are -37.563, -32.451, and -32.343 points. More second-value
strata do not yield monotone improvement. The predefined family is abandoned as
a competitive estimator; there was no search for corrections after these results.

The union-gap diagnostic is large: 3.911–10.142 points on the original and
26.453–32.088 points averaged across seeds. [Inferred] The rare-event union
approximation is poorly justified here. This does not isolate its error from frozen
gates or Gaussian mass outside the rank chamber. [Measured] Neither tested extra
stratification level fixes competitiveness.

The bounded run took 1.187 CPU seconds inside `main` (excluding interpreter/module
startup); single-thread limits were set before execution. Candidate runtime was
2.90–3.81 times sampling at the same FLOP cost, so the ledger needs DRI review before
headline use. Cached model-free precomputation was warmed by sanity checks. The
first restricted run stopped at checkpoint loading without test estimates; the
same script then completed with authorized public-checkpoint network access.

Exact reproduction command from repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/clean-neurons/.venv/bin/python m16/approaches/crossover_compute/candidate_experiment.py
```

`candidate_raw.json` preserves each estimate, exact cached truth and standard error,
actual cost, union gap, CPU time, unsupported-budget reasons, and estimator/truth-file
hashes. `candidate_table.json` contains all 18 group/level aggregates and uncertainty
limitations. Reproduction requires public AlgZoo checkpoint access and the repository's
Python dependencies; no new truth simulations are performed.

A possible future diagnostic, **not performed or needed to claim a win here**, is a
DEV-only comparison with joint, rank-respecting Gaussian failure integration to
separate union overlap from chamber leakage. It would need its own compute ledger
and task claim. This experiment does not establish that it would help.
