# Independent T264 fixed retained-panel audit

## Question
Does the proposed cost/error frontier follow from the single predeclared retained panel and its stated numerical-value/cost convention?

## Method
Checker a1b54d95 plus execution-log pin02f61e99 independently decodes IEEE754 bits into Fractions, validates NPY/JSON/provenance, and checks all stored frontier records using linear prefix sums, cross-multiplied ratio ordering, exact selected-index sums and expanded cubic decisions. Neither optimizer is imported or called. Source reviewer56 gate4a399be6/sharedb6e26acf preceded immutable gate09eafaf9/sharedeaa96dde and the sole saved audit. Producer3660020e, its preceding gated source, all raw outputs and earlier controls remain byte-bound. Assignment0c9121eb; reciprocal source review0f2bbeb7.

## Status
Complete/pass; no failed attempt. Only original H16T4 zoo0 chamber at45e1f6c was audited. Scientific interpretation remains conditional on retained numerical errors and inherited operation counts; no further panel or estimator execution is cleared.

## Results
[measured] All51,552 rows,22,521 positive-cost entries,22,522 fractional breakpoints, all three method prefix/selected-index records and80 boundary decisions pass. Seven producer/five reviewer events reconcile. Six NPY vectors and two numerical JSONs were independently decoded once. No invalid values or source-accounting mismatch caused a failure; inherited floating sum discrepancies remain in the exact validation record rather than being forced to zero.

[inferred from checked retained values] Original nominal cost is8,375,735,504 FLOPs. Fixed work outside the wrapped mass calls is411,949,282 nominal FLOPs; mass-call cost includes its own internal triangulation/scanning work. The checked integer-optimal remaining-cost bracket is approximately1,470,771,585.224 to1,471,873,760 nominal FLOPs (exact rational lower bound is retained in results/frontier; the displayed lower number is rounded). The upper endpoint is achieved by the checked ratio-prefix subset, not a claim that it is the integer optimum.

| Exact comparator | Selected rows | Remaining nominal FLOPs | Retained removed error, approximate |
|---|---:|---:|---:|
| Original-cost error prefix | 47,842 | 2,180,664,002 | 0.00002454912604 |
| Self-consistent error prefix | 49,081 | 1,597,294,956 | 0.00005619737401 |
| Self-consistent ratio prefix | 19,852 | 1,471,873,760 | 0.00005848469261 |

All exact feasibility checks use the fixed retained p and canonical H16T4 sampler cost2378 in Q=p(1−p)*2378/2. The source historical NumPy skip_at_1se report is retained separately; its absent selected identities, floating summation and tie order are not reconstructed or equated to the exact comparator. Float SD diagnostics have only the predeclared squared-residual rounding check; all exact frontier fields have zero tolerance.

## Compute accounting
[measured] Reviewer main CPU2.773304452s; producer main CPU3.644783240s is separate. Actual reviewer soft/hard CPU20/20 seconds and AS2147483648/2147483648 bytes are recorded. No helper/model/mass/geometry/RNG calls. Six retained arrays/two JSONs plus saved frontier decoding, hashing, rational arithmetic and serialization are real review work. Top-level imports/final serialization are outside main timing but inside OS caps. No native operation-price census or deployment construction cost is supplied.

Exact command used once; outputs refuse overwrite:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash -c 'set -e; set -o noclobber; ulimit -Sv 2097152; ulimit -Hv 2097152; ulimit -St 20; ulimit -Ht 20; flock /workspace/work/codex-review.lock timeout 25 /usr/bin/python3 /workspace/work/feynman_fundamentals/m16/approaches/feynman_fundamentals/t264_frontier_review/panel_saved/check.py > /workspace/work/feynman_fundamentals/m16/approaches/feynman_fundamentals/t264_frontier_review/panel_saved/run1.log 2>&1'
```

## Claims
[inferred] Within this fixed retained-value oracle problem, feasible unequal-cost selection can more than halve nominal cost while meeting its recalculated nominal sampling-SD condition. The fractional relaxation supplies an optimistic lower remaining-cost bound; the actual feasible subset supplies an opposing upper bound. This corrects a ceiling inferred from a single error-sorted heuristic, without validating true integration bias, original partition completeness or full native cost.

## Alternatives not ruled out
Discovering cheap error bounds or selecting cells may consume these apparent savings. Numerical errors may differ from true error masses; omitted operations and new geometry can alter costs. Other panels, integer optimizers or deployable constructions remain untested. No H16T10 scaling, noisy-model transfer, MSE/rung pass or algorithm-family ceiling follows.

## Failure modes
Negative errors/costs, noninteger costs or nonpositive fixed work violate this contract and must not be clamped away. Ignoring nonmonotonic feasibility, dropping zero-cost accounting, treating retained numerical masses as exact Gaussian masses, or claiming oracle selection is free would invalidate extrapolation beyond this audit.
