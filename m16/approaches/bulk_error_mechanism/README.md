# T116 bulk-error mechanism track

## Question
Which weight-derived mechanisms force bulk errors in original H4T4 seed0,
and can those families be counted more cheaply than detailed gate boundaries?
Claimf51b14a was pushed before implementation. This is the bulk-track T116,
not the duplicate triangle-fallback row; DRI renumbering is requested.

## Method
Cost-first readout reachability: h>=0, normalize sum(h)=1, and test each output
as first-index argmax with exact binary-rational weights. For j>0 maximize a
common positive margin over earlier indices while requiring weak margins over
later indices. Enumerate exact basic feasible points and derive a matching
nonnegative dual certificate; zero/negative optimum proves j cannot be selected
at nonzero h. h=0 always selects0. This is a superset of actual reachable states,
so feasibility does not prove the network reaches that label. Freeze zero,
duplicate and componentwise-dominated readout controls before trained evaluation.

Then reuse existing exact T68 H4T4 gate/correct simplex artifacts read-only,
not a new partition engine. On each simplex, exact rational interior recurrence
recovers the complete gate history and affine hidden/logit maps. Freeze mutually
exclusive gate-based families before numerical masses: final all-off; earlier
all-off reset; no reset but locally forgotten input in output contrasts; remaining
cells. Record ties and hidden-vs-decision forgotten columns as separate overlapping
diagnostics. Local derivative zero does not establish global input irrelevance.
Correctness simplices use the same classification; subtract correct mass from
family mass to attribute errors. Verify every full/simplex partition, exact
vertex gate consistency and known .477013125994119 total independently.

Start with reachability only; source/artifact inventory and cost forecast before
mass integration. Any approximate recipe or model expansion requires a separate
frozen discriminator. Coordinate worker34 T110 partition-cost work and use
positive simplex mass rather than inherited cheap orthant approximation.

## Alternatives and falsifiers
1. Missing output classes force a bulk floor: exact readout dual can refute or
support this without relying on truth. Feasible all-h witnesses are only a
necessary condition for actual reachability.
2. Entire-prefix resets cause most errors: classify using exact gate histories
and compare mass/error, not counts. Counts alone can mislead.
3. Apparent ties/forgetting are floating thresholds: use exact rational maps,
first-index ownership and independent direct checks.
4. Bulk errors remain despite no simple reset/forgetting: preserve residual
family, never force every error into a preferred narrative.

## Status
Protocol frozen; exact readout reachability first.

## Results
Pending. Original accuracy reference .477013125994119 is diagnostic only,
not fitted estimator input. No noisy-rung or full-challenge claim.

## Compute accounting
One thread,2GB, project locks, <=110CPU phases. Exact readout phase <=20CPU.
Charge source/artifact loading, rational feasibility/duals, classification,
masses and independent checks; historical reference construction is reported
as a cold dependency, never free per-model preprocessing. No sampling.

## Claims
[speculative] A few exact weight-defined bulk mechanisms dominate this model.
[inferred] Unreachable readout labels can give an irreducible error floor.

## Alternatives not ruled out
Actual hidden reachability restrictions, distributed gate history, nonzero
coalescence, exact rank-aware regions and general bulk-error approximations.

## Failure modes
Confusing local derivative zero with globally forgotten input, all-h vs reachable
hidden sets, non-strict first-index ties, overlapping category double counting,
expensive mass evaluation, and uncertified numerical/roundoff error.
