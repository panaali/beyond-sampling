# Fixed T266 finite-law numerical controls

Claim `213f4b30`; immutable written predecessor `197b7ee3`, independently
`c01beb62`. T55 reviews this source/resource contract; recover_78 independently
audits saved results. Neither review may be inferred from the written predecessor.

## Frozen experiment

Run exactly 13 rational finite-law cases, six strict/tied label controls and 12
guard rejections once. Inputs and independent expected scalars are literal in
`fixtures.json`. No model, training, Gaussian/CDF, RNG, input sampling, quadrature,
checkpoint or geometry calls occur. Table correctness/surrogate values are given
synthetic data; no cost of obtaining trained values is claimed.

Reversal is defined by reversing each integer state tuple. The table must contain
distinct states, be closed under reversal and have positive rational weights
summing to one. Correctness is an integer 0/1, surrogate lies in [0,1]. Exact
Fraction arithmetic computes residuals, reversed and paired residuals, means,
absolute envelopes, cancellation, variances, covariance and budget-floor ratios.
All returned fields, including each row's residual, are saved before comparison.

For an invariant law, residual and paired means agree. The exact absolute gain is
the weighted minimum magnitude on opposite-sign pairs. A symmetric residual has
no absolute gain; a one-sided residual has no absolute gain even when pairing
reduces sampling variance. The covariance identities are tested separately.

The deliberately non-invariant table opts out of the invariant-law requirement
and reports the failed mean identity rather than asserting it. General mean and
variance formulas still use both marginal means/variances. A separate strict-law
guard must reject that same table. Non-invariant conditional laws are not promoted
to unconditional Gaussian pairing identities.

The finite table may place positive mass on fixed points or on endpoint-equality
states. The lexicographic orbit representation handles fixed points once and
two-element orbits with both weighted residuals. The usual x0<x_last half-domain
formula is asserted only when endpoint equality has zero mass and the law is
invariant. The fixed-point and moving endpoint-tie cases retain this distinction.

Label controls recompute the second-largest input index using decreasing value,
then increasing index. They compare its reversed index to the nominal relabeling
and retain exceptions on ties. Output winner uses first-index max, including
negative and tied logits. These labels are synthetic, not network evaluations.

## Fixed expected cases

The 13 names fix antisymmetric full cancellation; constant positive residual;
optimistic and pessimistic one-sided residuals; symmetric nonconstant residual;
zero covariance; unequal invariant orbit weights; a non-invariant conditional
law; a positive-mass fixed point; endpoint-tied nonfixed points; a tripled paired
price; a budget-floor difference; and unequal fractional residual magnitudes.

Single/paired prices and budget are declared hypothetical rational scalars, not
measured algorithm costs. With m=floor(B/cs), n=floor(B/cp), report
m*Var(pair)/(n*Var(single)); zero single variance yields null rather than a ratio.
The tripled-price and B=5 cases distinguish this from the asymptotic 1+rho rule.
None of these tables establishes a deterministic quadrature or trained saving.

## Failure and resource contract

`launch.sh` imposes soft/hard CPU 20/20 seconds, address space 2 GiB, one-thread
environment and 25 wall seconds after the shared review lock. Driver checks actual
limits and invoked/resolved stdlib Python; its internal CPU cap is 17 seconds.
The main timer includes imports after its start, pins, reads, contractions and
most serialization; initial imports and final save can fall outside it. External
limits cover the whole process. No automatic retry, cap increase or changed
fixtures/tolerances is allowed. There are no floating scientific tolerances.

The driver requires published hexadecimal source-review citations and verifies
the six direct source pins, completed prior proof review and every inherited pin.
The immutable execution gate must be published before launch. Fresh output paths
refuse overwrite. Atomic writes preserve returned records before assertions and
progress/cap checks; exceptions set complete/pass_all false and retain traceback.

Expected successful coverage is 25 analyze attempts, 13 returns, 12 deliberate
ValueError rejections, six label calls and 64 progress events. Valid tables contain
37 rows and 37 reverse-residual lookups. These are structural research-work counts,
not a complete integer/GCD/native operation ledger. Decode, validation, exact
arithmetic, labels, recording and reviewer CPU remain charged as research work.

## Scope after completion

An independent saved audit must check all finite-law moments, expected fields,
label ownership, guard inputs, pins, resource records and progress without calling
this primitive. Passing closes these fixed controls only. A trained pairing test
still needs ONE frozen deployable surrogate, its justified integral, a feasible
paired representation and complete two-evaluation/discovery/refinement costs.
No trained scope or surrogate is silently selected here.
