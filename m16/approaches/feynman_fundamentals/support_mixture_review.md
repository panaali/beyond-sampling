# T40 support mixture and bounded Gaussian correction

T55 independent read-only review, 2026-10-17. This extends the cumulant check in 66efd3e and the real Gaussian bulk-integration failure preserved in 6e6ead8.

## Constant winner and mixture weights

On a cone where logits are linear, a candidate wins throughout its relative interior if every competitor margin is nonnegative on every generating ray, and every earlier-index competitor has at least one strictly positive ray margin. A nonzero nonnegative linear functional is positive on the relative interior; its zero face has zero continuous cone mass. An identically tied earlier competitor must instead win. This proves the real-arithmetic criterion used in `gate_mixture.py`; floating ray construction and margin signs are not certified by it.

The gate/rank components already have joint Gaussian masses. Aggregation is their mass times the endpoint probability of that component's target label. There is no additional division by three. Constant components use the exact winner indicator. Residual conditional moments divide by their own numerical mass, while aggregation uses the separately computed geometric component mass; the saved discrepancy is correctly retained.

The independent Decimal reconstruction in `check_support_mixture_review.py` imports no endpoint or geometry code. It finds 38 components, 25 constant, total mass 0.999999999999961801260, and constant-component mass 0.97082139088407668568. All five raw endpoint aggregates match the saved results within 3e-17. Matching residual components by gate/order key gives bounded full-covariance aggregate 0.4090643340795151832, matching the saved 0.4090643340795152. CPU: 0.00211 seconds. This verifies weighting, not the endpoint approximation itself.

## Corrected Gaussian integral

`full_covariance_bounded.py` fixes the independently reproduced missed-bulk failure of the original unbounded adaptive quadrature. Intersecting the integration domain with [-12,12] loses at most 2 Phi(-12), because the integrand is bounded by the standard normal density. Explicit bulk breakpoints and the conditional CDF transition address the reported failure. A lower endpoint beyond 12 can return zero within that tail allowance. Zero-variance covariance rows are now rejected unless all their entries vanish. Strict first-index ties are handled on deterministic differences.

The near-perfect-correlation error bound acos(abs(rho))/(2*pi) follows by integrating the Plackett correlation derivative bound. It applies to valid correlations; an excursion outside [-1,1] corrected numerically is explicitly flagged and has no certified covariance correction bound. Adaptive quadrature error estimates remain diagnostics rather than rigorous enclosures. The small trained aggregate bias and fixed support mass do not establish a universal Gaussian endpoint or a cost win. Original defective files and finite control failures remain preserved.
