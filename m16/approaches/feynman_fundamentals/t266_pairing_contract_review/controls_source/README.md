# T266 finite-law pairing controls source/resource review

## Question
Do the fixed rational tables and label/guard cases faithfully exercise the already reviewed pairing identities without implying an unimplemented trained estimator?

## Method
Read primitive.py, controls.py, fixtures.json, PROTOCOL.md, protocol.json and launch.sh. Check source bytes and predecessor proof pins without importing the primitive or executing the numerical cases. Fixture inspection is literal source inspection; independent saved arithmetic follows the sole producer run.

For R reversing a state tuple and r=Y-S, the source computes r_R, p=(r+r_R)/2 with the original table weights. Without invariance, E[p]=(E[r]+E[r_R])/2 and Var(p)=(Var(r)+Var(r_R)+2Cov(r,r_R))/4; it correctly retains both marginal moments. Pointwise (|r|+|r_R|)/2-|p| equals min(|r|,|r_R|) on opposite signs and zero otherwise. Under invariant weights this becomes the reviewed absolute-gain identity and E[p]=E[r]. No change of signed mean or automatic absolute improvement is claimed for one-sided residuals.

Lexicographically chosen orbit representatives count fixed points once and two-state orbits with both original weighted residuals, even under a non-invariant law. The x0<x_last half-domain is asserted only with invariant weights and zero endpoint-tie mass. Endpoint equality can contain nonfixed reversal orbits; the explicit four-coordinate tie fixture preserves this distinction.

The finite-budget variance ratio is m*Var(p)/(n*Var(r)), m=floor(B/c_single), n=floor(B/c_pair). It is a variance ratio for hypothetical independent samples of each type, not an MSE ratio when their means differ and not a measured cost. Zero original variance returns null. The non-invariant fixture deliberately returns paired mean0 versus residual mean-1/2 and surrogate-plus-pair3/4 versus accuracy1/4; its zero paired variance must not be advertised as an unbiased accuracy advantage.

## Status
Source/resource gate passes for61ca7042/sharedda40a449 under assignment213f4b30. Published completed source gate and immutable execution binding are required before the sole run. The inherited written review c01beb62 was executionfalse and remains unchanged; this new gate covers only these finite controls.

## Results
[inferred] Literal expectations cover antisymmetric cancellation, constant and one-sided residuals, symmetric/no-gain covariance, unequal orbit weights, non-invariant conditional law, fixed/tied orbits, fractional residual magnitudes and budget floors. In particular the same zero-covariance table has ratio1 at doubled cost,3/2 at tripled cost and5/4 at budget5 with doubled cost. These are source-level algebra checks, not executed results.

The six label controls use decreasing value then increasing index for the second-ranked input and first-index logit max. Strict rank reversal relabels correctly; all-tied-four, two-maxima and palindromic maxima retain explicit counterexamples. All-tied-three is a valid exception where the mapped middle index agrees. None is a network evaluation.

## Compute accounting
Thirteen valid tables contain37 rows and37 reverse lookups;12 rejecting guards produce25 analyze attempts and13 returns; six label attempts/returns. Fixed event structure is one prerequisite event, two per31 cases/labels/guards, and one complete event:64. Structural counts omit individual Fraction/GCD/sort/validation operations; full research CPU remains charged.

Launcher uses review lock, soft/hard20/20CPU,2GiB,25wall after lock and single-thread environment. Driver verifies actual runtime limits, invoked/resolved stdlib Python, direct source/reviewer pins and all inherited proof pins. Internal17CPU checks follow flushed progress. Fresh output directory and noclobber log prevent overwrite; returned input/output records precede assertions, exceptions reset complete/pass flags and retain traceback, no automatic retry.

## Claims
[inferred] The fixed source suite tests finite-law identities and bookkeeping under its stated premises. It does not establish Gaussian numerical performance, a deployable surrogate integral, a paired geometric representation, deterministic quadrature improvement or a trained-model speedup. Separate independent saved review is required.

## Alternatives not ruled out
Pairing may help absolute error for sign-opposed residuals but cost more; one-sided pairing can improve sampling variance without improving the absolute envelope. Non-invariant conditional restrictions require a new law argument. Real two-evaluation/refinement costs may differ substantially from declared table prices.

## Failure modes
Positive-mass ties/fixed points invalidate an unqualified half-domain reduction. Mean-shifted pairs invalidate an unbiased accuracy interpretation. Fixed guard coverage is not exhaustive arbitrary-input API validation. Passing these controls does not select or authorize a trained pairing variant.
