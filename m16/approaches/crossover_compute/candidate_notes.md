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
