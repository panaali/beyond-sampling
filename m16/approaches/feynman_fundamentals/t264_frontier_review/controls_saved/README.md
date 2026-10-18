# T264 independent saved synthetic-control audit

## Question
Do the fixed synthetic frontier certificates match independent exact arithmetic, their frozen fixtures, and execution records?

## Method
`check.py` uses direct prefix subset sums, cross-multiplied ratio order, expanded cubic midpoint decisions, exact declared optima and boundary derivatives. It checks every comparison method and invalid-input predicate without importing or calling the producer primitive. Source6771bd9d was independently gated by56 at9bc7b9ea/shareda211efc7; immutable gate64f470e1 was published7072212b before execution. Producer rawb69fecb2 and its preceding gateabc42874/sharedd03a928e are retained and byte-bound.

## Status
Done: sole saved audit passed; no failed reviewer or producer attempt in this bounded phase. Scientific acceptance remains with the DRI. No retained-panel execution is authorized by this result.

## Results
[measured] All12 exact cases,6 invalid guards,26 breakpoints,560 saved midpoint decisions, all feasible-prefix selections,18 attempts/12 returns,38 producer events and20 reviewer events pass. The unequal-cost case retains savings0/0/100 for fixed-error/self-consistent-error/ratio prefixes. The nonmonotone case retains an infeasible interior point and a feasible full endpoint. There are no numerical tolerances.

Exact command used once (outputs refuse overwrite; do not rerun against existing outputs):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash -c 'set -e; set -o noclobber; ulimit -Sv 2097152; ulimit -Hv 2097152; ulimit -St 20; ulimit -Ht 20; flock /workspace/work/codex-review.lock timeout 25 /usr/bin/python3 /workspace/work/feynman_fundamentals/m16/approaches/feynman_fundamentals/t264_frontier_review/controls_saved/check.py > /workspace/work/feynman_fundamentals/m16/approaches/feynman_fundamentals/t264_frontier_review/controls_saved/run1.log 2>&1'
```

## Compute accounting
[measured] Reviewer main CPU0.032279629s; producer main CPU0.076398697s is separate. Actual reviewer soft/hard CPU20/20 seconds and address space2147483648/2147483648 bytes are saved. Timeout25 seconds follows lock acquisition. Hashing, rational arithmetic and serialization are review overhead, not free estimator advice. Stdlib startup and final serialization are outside the saved main timer but inside OS limits. No producer primitive calls, target payload decoding, model, geometry, mass, CDF or RNG calls. Full arithmetic/bit complexity is not priced.

## Claims
[inferred] These exact fixed synthetic outputs implement the reviewed fractional-envelope and prefix-comparator semantics consistently. This is a finite saved-data check, not a proof of all implementation branches. The written no-interior-minimum argument supplies the global-boundary interpretation.

## Alternatives not ruled out
Other fixtures can expose defects; retained numerical masses and inherited operation costs can differ from true quantities; ratio-prefix savings need not be the integer optimum. No native-cost, true-mass or deployable estimator result follows.

## Failure modes
Zero or negative fixed remaining cost is outside the contract. Feasibility is not globally monotone. Zero-cost positive-error rows must not be forced into the fractional envelope. Any future storage/performance rewrite needs its own frozen source and controls before retained-panel use.
