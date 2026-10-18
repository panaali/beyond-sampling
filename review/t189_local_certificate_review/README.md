# T189 local certificate independent review

## Question
Does the proposed affine-interval recurrence and Gaussian rank partition yield valid accuracy bounds?

## Method
Read THEORY.md and the fixed staged plan. theorem_gate.json records an independent algebraic argument and exact source pins.

## Status
Theory passes after the conservative crossover wording correction. Implementation, controls and application remain separately gated.

## Results
Affine and ordinary ranges can safely be intersected. The three fixed ReLU slopes preserve enclosures without remainder independence. Conditional exchangeability supplies rank-event mass1/m; whole-box output certificates avoid assuming output/rank independence. Tails remain unresolved. The abstention lower bound only gives a sufficient actual-width-versus-sampling-SD comparison.

## Compute accounting
Text reads and hashing only; zero model, RNG, fitting, geometry or scientific numerical calls.

## Claims
[inferred] The mathematical construction is sound under exact predicates and directed probability arithmetic. No implementation or trained numerical result is asserted.

## Alternatives not ruled out
Finer domains, other valid slopes, reachable-state constraints, conditional integration and cross-region aggregation remain open.

## Failure modes
Undirected CDF normalization, omitted tails, first-index tie mistakes, rank/output independence assumptions, and confusing uncertainty width with estimator error would invalidate stronger conclusions.
