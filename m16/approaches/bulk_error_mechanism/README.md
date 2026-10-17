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

## Readout reachability result
All four H4T4 labels have exact nonnegative-state witnesses with positive
margins (optima .30888146,4.57925045,2.97702237,1.09322867 in the stated
first-index LP). Each optimum has a matching rational nonnegative dual bound.
Thus the simple globally missing-readout explanation is falsified: it certifies
zero irreducible error here. These witnesses need not be attainable by the
actual recurrence; hidden reachability restrictions remain open.

Zero/duplicate/componentwise-dominated controls allow only label0; identity
readout allows every label. Full720 rational solve attempts cost .154739CPU
including compressed artifact loading,56MB RSS. `readout.json` preserves all
exact primal/dual coefficients and source hashes. Reproduce `python readout.py`
under the review lock. Independent certificate review requested before any
mechanism conclusion; no numerical tolerance defines reachability.

## Frozen artifact attribution and numerical gate
T68 stores3853 full-partition and5058 correct-event simplices, not merely
floating gate samples. Reconstruct exact gate history and maps at a positive
integer interior combination, then verify every simplex ray against each exact
gate inequality. Use common dyadic denominator scaling to keep recurrence/map
arithmetic integer. Correct-simplex interiors must directly predict their
second-ranked input; first-index ties are explicit. Classification priority is
final all-off, any earlier all-off, locally zero output-contrast column without
reset, remaining. Tie-pair and forgotten-column diagnostics are recorded
separately, not added as overlapping error categories.

No new geometry enumeration. Existing positive mass rules16/20 are run on
both artifact lists, with per-category numerical bounds and total/reference
checks. Gate total must be1 and correct total match.477013125994119 within
combined integration bounds plus1e-11 arithmetic allowance. Compare rule
changes separately; numerical arithmetic remains uncertified. CPU cap40seconds,
no noisy copies. This is diagnostic exact-geometry attribution, not a cheap
production estimator. Report prior construction cold dependencies explicitly.


Independent T55 review reproduced all20 saved primal/dual certificates using
Fraction arithmetic without importing the author LP solver (.063CPU). This
checks exact feasibility, normalization, stationarity, signs and equal
objectives. It confirms only the readout-superset statement. The generic
routine currently raises if normalized weak-later constraints are infeasible,
rather than returning an infeasibility certificate; that unsupported case
occurs in none of the frozen rows. Do not call this a complete general LP
classifier or infer actual recurrent reachability from these witnesses.


## Exact-history bulk attribution result
All8911 exact simplex interiors and570304 vertex/gate inequalities pass direct
integer checks. The3853 gate and5058 correct simplices share450 distinct gate
histories. Positive mass GL20 totals are.9999999999999993 and
.4770131259941177, agreeing the independent reference. GL16/20 group changes
are at most3.5e-14; conditional quadrature bounds, not floating certificates,
remain attached to every category.

| Weight-defined family | Probability mass | Correct mass | Error mass |
| --- | ---: | ---: | ---: |
| Final all-off | .0253699870522870 | .0030455804943812 | .0223244065579058 |
| Earlier all-off | .0009063207394358 | .0005285810685598 | .0003777396708760 |
| No reset, exact contrast forgetting | 0 | 0 | 0 |
| Remaining | .9737236922082765 | .4734389644311767 | .5002847277770999 |

The remaining category contains95.66% of all errors. Simple exact resets or
exact locally forgotten input columns therefore do not dominate this model's
bulk error. This is a negative about those precise mechanisms, not approximate
memory loss, support-dependent readout reachability, or a broader mechanistic
recipe. Input influence can be very weak without being identically zero.

The first run completed all classification but hit Python's4300-digit decimal
serialization limit on an exact determinant accumulator. The traceback and
cost are preserved. Hexadecimal numerator/denominator storage repairs only
serialization; geometry and classification did not change. `attribute.py`
reproduces the completed result under the primary lock. All source hashes,
per-simplex history assignments, exact category determinant sums and numerical
bounds are saved. Original T68 geometry construction is a historical cold
reference dependency, not free preprocessing in a proposed estimator.

Completed attribution cost 6.239897062 CPU seconds including imports, RSS 65808 KiB. Failed-run CPU is additional and saved separately. This is a diagnostic, not an efficient estimator. Independent T55 source/attribution review is requested.
