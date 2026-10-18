# T209 conservative cap utility on saved exact-reference cones

## Question
Are the newly safe cap bound and a saved-certificate determinant bound tight
enough to merit a later pruning-adoption question on this fixed diagnostic set?

## Method
Reuse64 original T193 pilot rays and exact KKT certificates, unchanged T200
cap source, and saved numerical references. THEORY.md derives the determinant
bound; PROTOCOL.md freezes scope and per-cone thresholds. No integration or
geometry replay; no kernel/cost-chain continuation.

Two competing explanations are deliberately separated: cap width may ignore
thin sliver geometry, while even a tighter certificate-based bound may be
expensive to construct. The fixed same-ray comparison diagnoses the former;
existing construction ledgers remain visible for the latter. No production
cost conclusion is authorized by this diagnostic.

## Status
Fixed64 phase c9a03d9d completed after shared immutablegate9d5faa2b;
source/theory review1a1ea51f passed. Independent saved audit39b10901/shared732867a4 passed all64 rows,96 archives,130 progress records and exact ledgers/contractions. This bounded phase is complete; DRI adoption remains separate.

## Results
All64 completed with no cap fallback or conditional-reference disagreement.
Outward cap/stored-reference ratios: min80.30594, median6655.29544, max2.7774e7.
Determinant/KKT upper ratios: min1.096624, median1.097814, max2.254136.
At frozen thresholds1e-4 through1e-8, cap counts36/26/16/5/1 and determinant
counts64/62/59/50/33. All indices/bounds remain in results.json. These descriptive
ratios use conditional numerical references, not complete floating certificates.
No extrapolation, threshold/axis change or production winner is selected.

## Compute accounting
One process:0.750322521 recorded CPU measured inside main, including loading and repeated durable
serialization but excluding imports and final result serialization.64 cap calls take0.098152403CPU;64 saved exact determinant
contractions0.000323590CPU. Cap ledger9472 macro steps, peak346 fraction bits;
big-integer/control/native work excluded. Zero model/geometry/KKT-solve/mass/
reference calls, under20CPU/2GB review lock. Prior certificate construction
and validation inventories retained separately, not free production work.

## Claims
[inferred, independent source/theory review1a1ea51f] Both bounds enclose represented cone mass under the
stated exact-ray/certificate assumptions. [speculative] One may be useful for
pruning; practical tightness and implementation overhead remain unresolved.

## Alternatives not ruled out
Other axes, exact spherical bounds, cheap outward floating implementations,
other simplex bounds, lower-dimensional or class-restricted pruning. No finite
64-cone diagnostic proves general failure or coverage.

## Failure modes
Changing ray normalization without updating certificates; treating conditional
references as exact; converting a bound downward; hiding fallback1; counting
already-built exact certificates as free; confusing cone mass with total
classification error or using per-cone thresholds as a global budget.

## Interpretation and untried rescue
This fixed cap is safe but loose relative to the saved references. The two
bounds differ in both determinant-volume information and norm lower bound, so
this comparison does not isolate a unique cause. Existing T130/T161 center-
Cauchy bound can replace KKT with a positive-axis lower norm. Another exact
alternative uses the positive minimum Gram entry as the simplex norm lower bound. T211 records a
blocked exact/outward follow-up, not a newly invented theorem. No additional
computation or revival of the frozen native-cost chain is authorized here.

Command: `python review/cap_saved_utility/run.py`, one-thread review flock,
external20CPU/2GB; outputs refuse overwrite. Source/protocol/reviewer gate was
published before execution. All130 progress records and run1.log retained.
