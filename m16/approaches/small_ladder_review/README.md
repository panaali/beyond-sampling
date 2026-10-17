# T52: independent review of the small-model ladder

## Question
Do the small-model candidates calculate accuracy correctly, including positive-mass output ties, and beat sampling under a defensible compute comparison?

## Method
Root owns the review while six workers research their assigned cases. The earlier unpushed reviewer files were absent after the workspace restart; this is a new review record, not a claim that those artifacts survived.

Freeze the next checks before execution:
1. Compare H2,T3 trained and independently perturbed models against the octant-first enumerator in `facet_transport/probe.py`. The candidate uses the independently developed two-input tail. Check all-zero and duplicate-row readouts separately.
2. Audit the operation-count convention and compare runtime against a vectorized sampling baseline separately. Kernel-call counts alone are not forward-equivalent compute.
3. Review longer-case geometry and integration bounds before accepting their nominal estimates. A convergence check alone is not a proof of floating-point geometry correctness.

### Alternatives and cheapest falsifiers
- A seemingly exact estimate omits a positive-mass tie region: use all-zero and duplicate-row output models with independently known first-index predictions.
- Both implementations share a geometry bug: use a closed-form nonrecurrent H1 family as an independent calibration and inspect the clipping conventions.
- Apparent compute advantage comes from omitted preprocessing: inventory geometry, numerical integration, and per-model setup; report wall time separately from arithmetic cost.

## Status
Active. No new candidate independently accepted yet. Claude remains responsible for project-level confirmation.

## Results
Pending recovery-cycle checks. Existing T55 conditional-primitive reviews are preserved in its own folder and do not establish a complete estimator.

## Compute accounting
Numerical checks acquire `/workspace/work/codex-compute.lock`, use one thread, bounded chunks, and the repository memory limit. Validation sampling is reported separately from estimator cost.

## Claims
- [measured] The recovered repository contains the committed T55 primitive and its bounded prior reviews.
- [speculative] The T3 estimator may provide another complete small-model benchmark; current validation is pending.

## Alternatives not ruled out
Numerical clipping error, correlated validation implementations, unsupported operation proxies, and model-specific shortcuts remain possible.

## Failure modes
Thin regions, nearly duplicate output rows, high-degree moment cancellation, and missing compute charges can each invalidate a result.
