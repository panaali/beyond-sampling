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

## Frozen strongest untried rescue: constant-output whole gate cones
Exact contrast-column erasure is stronger than output-label insensitivity.
An input can change all logits while leaving the selected class fixed. Before
retiring an output-forgetting explanation, group the existing full-partition
rays by450 gate histories and reconstruct their exact final logit maps. For
each candidate label, verify every competitor margin on every cone ray. A
margin identically zero against an earlier index rejects that candidate;
otherwise nonnegative rays and positive interior margin certify first-index
winner throughout the full-dimensional gate cone. Do not infer a whole-cone
property from a single triangulation piece or center.

Predeclare two classes among the previously remaining histories: certified
constant output on the entire gate cone; decision-varying or not certified.
Report all labels and exact output-contrast ranks, then use unchanged GL20
mass on the same gate/correct artifacts. No model perturbation, fitted threshold,
new cone partition or truth-selected recipe. Verify certificate margins by
independent integer source checks and preserve known totals. <=20CPU review
lock, one thread. This diagnoses a plausible discrete-output mechanism despite
nonzero derivatives; it does not itself remove enumeration cost or prove global
state forgetting across distinct gates.

### Constant-output rescue result
The stronger test certifies166 additional no-reset gate histories whose argmax
is constant on the full-dimensional cone interior (boundaries may have different
first-index tie ownership). Their mass is.2746298311839157 and error contribution
.13511906916754257. Thus exact contrast-erasure would have missed a substantial
discrete-output family: nonzero logit derivatives do not imply changing labels.

However267 remaining histories, mass.6990938610243618, still contribute
.36516565860955724 errors. Their selected class varies inside the gate cone;
constant-output gate interiors plus resets explain only about30.18% of error.
This rejects their sufficiency as the whole explanation on the fixed model,
not their value as components or the possibility of coarser cross-gate regions.

All450 exact logit maps have contrast ranks134 at rank3,222 at rank2,81 at rank1,
and13 at rank0. All four actual classes now have recurrent cone-interior
witnesses via nonempty certified constant-output cells, stronger than the earlier
unrestricted-hidden feasibility check. No label is globally missing on this
model. The certification uses51,024 integer ray-margin checks; unchanged positive
GL20 mass attribution takes1.934568CPU including setup,66MB RSS. It retains the
original total/reference and conditional mass bounds. Reproduce
`python constant_output.py` under the review lock. Independent certificate/source
review requested; no portable-cost advantage or cross-model extrapolation.

Untested rescues remain explicit: overlapping gate cells may merge into simpler
common-output regions; weak but nonzero input influence may admit a bounded
approximation; exact rank boundaries can interact with internal readout regions.
Those require concrete cost/accuracy discriminators, not relabeling the residual
as unexplained noise or sweeping parameters until a favorable result appears.

## Frozen whole-cone correctness discriminator
To test whether bulk errors can be counted without internal readout/rank
clipping, derive conservative possible-label sets from every exact gate ray.
Exclude output j if one competitor k dominates it on the cone interior:
all margins L_k-L_j nonnegative and either k<j or some margin strictly positive.
Exclude true second-rank j if at least two inputs are strictly above it over
the full cone interior, or no other input can ever be above it. Strict interior
ordering means all ray differences nonnegative and at least one positive;
possible-above needs at least one positive ray difference. These tests preserve
supersets even though some retained labels may actually be infeasible.

Disjoint possible-output and possible-second sets certify the whole cone
uniformly wrong; matching singleton sets certify uniformly correct. Everything
else remains unresolved. Apply to the same450 histories and unchanged GL20
mass; validate certified-correct gate mass equals its stored correct-event mass,
and certified-wrong correct-event mass is zero, within numerical qualifications.
This is an independently falsifiable sufficient certificate, not exact feasible
label enumeration. Cost <=20CPU, no new region engine, truth fitting or noise.
Report event coverage and residual interval width before suggesting efficiency.

### Whole-cone correctness result and scoped negative
The sufficient certificate labels48 gate histories uniformly wrong and39
uniformly correct, but their total mass is only.005750412999... . Uniformly
wrong mass is.00284266898053852; uniformly correct mass is.00290774401952806.
All correct-event simplices avoid certified-wrong histories, and the certified
correct full/correct mass agrees exactly at recorded precision. This passes
the independent stored-event consistency controls.

The remaining363 histories carry.9942495869999322 of the input distribution.
Consequently this coarse certificate only gives the uninformative accuracy
interval[.0029077440,.9971573310], before numerical allowances. A large fraction
of certified history counts does not imply useful mass coverage. The tested
whole-gate output/rank-set shortcut is insufficient as an estimator; its
sufficient labels deliberately retain correlations and feasible possibilities
conservatively, so this is not a proof that stronger cone certificates fail.

The run costs2.417973CPU/66MB,102048 integer plane checks plus unchanged mass
rules. `cone_correctness.json` preserves every certificate/exclusion and all
mass bounds. A stronger plausible alternative is conditioning on ordered top-two
families before certification, as worker34's distinct T110 engine already does.
Do not duplicate that engine or retire coarse correctness certificates based
on this raw-gate control. Exact simultaneous rank/readout feasibility and
cross-gate merging remain untested; none is claimed impossible.

Independent T55 direct integer recurrence checks all450 saved constant-output
classifications on4252 unique per-history input rays without importing the
author finalmap/margin functions. All183 constant histories (including resets)
and267 varying histories agree; positive sums of rays provide genuine input
witnesses for each label0..3. Review CPU.236seconds, no geometry/mass rerun.
Thus actual recurrence reachability is established for all four classes of
this one checkpoint, while the original all-h LP alone did not establish it.
The whole-correctness shortcut remains separately queued for independent review.
