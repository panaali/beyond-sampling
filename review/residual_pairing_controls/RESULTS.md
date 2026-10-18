# Fixed exact numerical results

Sole producer run `81149d17` passed all 13 finite-law tables, six label fixtures
and 12 guards. The source/resource gate `e3456dfd`, shared `f223cbc0`, and immutable
execution gate `9b98e185`, shared `3a23fa9a`, preceded execution. Independent saved
audit `35ec8444`, shared `6906472a`, checks every exact field and provenance chain
without author calls (0.03293 CPU seconds, 32,464 KiB). No failure, retry or target call.

The 37 exact table rows give the following discriminators. Costs are fixed
hypothetical prices; variance ratios compare floor-budget sample counts and are
not measured algorithm performance or deterministic integration gains.

| Case | Residual mean | Absolute bound before → paired | Budget variance ratio |
|---|---:|---:|---:|
| Antisymmetric | 0 | 1 → 0 | 0 |
| Symmetric nonconstant | 0 | 1 → 1 | 2 |
| Optimistic S=1 | -1/2 | 1/2 → 1/2 | 0 |
| Unequal invariant orbit weights | 2/3 | 1 → 2/3 | 4/5 |
| Unequal fractional residual magnitudes | 1/8 | 3/8 → 1/8 | 0 |
| Zero covariance, paired price 2 | 0 | 1/2 → 1/2 | 1 |
| Same table, paired price 3 | 0 | 1/2 → 1/2 | 3/2 |
| Same table, price 2 and budget 5 | 0 | 1/2 → 1/2 | 5/4 |

The non-invariant table has residual mean -1/2 but paired mean 0. Its surrogate
mean plus paired mean is 3/4 whereas true synthetic accuracy is 1/4. Zero paired
variance therefore does not make this an unbiased accuracy estimator or an MSE
improvement. Strict invariant-law mode rejects the same table.

A finite law with positive mass on reversal fixed points needs that mass included
once. The x0<x_last half-domain integral misses it; lexicographic orbit integration
preserves the mean. Endpoint-equal but nonfixed states demonstrate a distinct
half-domain omission. Under the continuous Gaussian law these endpoint sets have
zero mass, but they cannot be dropped from these finite fixtures.

Strict-rank input labels obey the nominal reversal permutation. Tied controls
retain the failures and the all-equal T=3 exception; all output ties keep the first
index, including negative logits. There is no network evaluation behind these
synthetic correctness values.

Coverage is 25 analyze attempts, 13 returns, six label calls, 12 deliberate
ValueError rejections and 64 progress events. The process used 0.058386953 CPU
seconds and 19,380 KiB peak RSS, with actual 20/20 CPU seconds and 2 GiB limits.
No model, Gaussian/CDF, RNG, geometry, fit or checkpoint calls occurred. Exact
arithmetic/validation/serialization are research work, not free estimator advice.

This finite implementation phase does not choose or validate a trained surrogate.
The next trained discriminator still requires ONE frozen deployable S, its
justified integral, a feasible paired partition and complete discovery/refinement
and two-evaluation costs. S=1 supplies a narrow negative control; it is not a
candidate that demonstrates useful absolute-bound cancellation. Other surrogates
remain open.
