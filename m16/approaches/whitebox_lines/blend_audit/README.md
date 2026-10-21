# Independent saved-prediction ensemble screen

## Question

Could a linear combination of the already evaluated e30 and logxio predictions
justify paying both estimators' complete FLOP costs?

## Method

Preregistered before arithmetic: use the three common saved rows mini_0,
full_0, full_1. Match reference/model seeds and weight hashes. Compute final-layer
error second moments and cross moments. Evaluate a fixed 50:50 blend and the
unconstrained least-squares oracle blend, both per row and pooled across rows.
Blend is `a*e30+(1-a)*logxio`. Charge the sum of the original full FLOP costs;
arithmetic overhead is omitted, making the diagnostic slightly optimistic.
Oracle coefficients use the target itself and are not deployable estimates.

Competing explanations: shared errors leave little complementarity (test error
cosines); complementary errors allow enough cancellation to pay doubled cost
(test oracle adjusted error against e30). This cannot evaluate shared computation.

## Status

Complete independent arithmetic check under the parent task's authorization.
No full-cost ensemble is justified by this three-row diagnostic.

## Results

Saved in [results.json](results.json). Reproduce with
`/tmp/whest-env-t473/bin/python m16/approaches/whitebox_lines/blend_audit/audit.py`.
Errors are strongly aligned. Even fitting a separate ideal coefficient to each
row cannot recover the cost of running both estimators. A common oracle-fitted
coefficient nearly doubles the mean adjusted error relative to e30. This is a
negative diagnostic for this full-cost combination on these rows, not a general
exclusion of ensembles or shared-computation hybrids.

## Compute accounting

No estimator/model runs. Three pairs of saved 1024-vectors, one thread, negligible
memory and CPU. Complete original costs are charged to the hypothetical ensemble.

## Claims

- [inferred] An oracle improvement is an optimistic ceiling on this exact
  scalar linear-blend family, not evidence of a deployable improvement.
- [measured] The fixed and oracle blends lose to e30 on the saved rows after
  charging both complete costs. No new estimator execution was needed.

## Alternatives not ruled out

Shared computation, feature-conditioned combinations, retraining corrections,
and different estimators are outside this screen.

## Failure modes

Targets have finite Monte Carlo noise; three rows are development data; oracle
selection overfits. Runtime and peak memory of an actual ensemble are untested.
