# T67: exact penultimate-target terminal cache

## Question
Does removing final-stage interpolation for the penultimate target resolve
its measured large component bias, and how does that affect total accuracy?

## Method
For fixed incoming h_(T-2), insert target u then fresh scalar x. Partition
u at first-step ReLU roots. On each interval the final preactivation is
affine in (u,x). Enumerate final masks and original gate/readout halfspaces;
verticalize them with x support[-6,6] and x below/above u. Readout labelT-2
must strictly beat earlier indices and weakly beat later ones. Use half-open
first-u panels so shared boundaries own a conditional slice only once; zero
gates have one inactive branch. Cache active affine lower/upper x bounds
and their complete u feasibility/intersection endpoints. Stable normal CDF
differences give exact scalar mass in real arithmetic.

Reuse the prior original-halfspace verticalizer and query evaluator locally;
no shared/canonical edits. For the tested H2T4 base, start the label2 backward
recurrence at this exact stage2 cache, bypassing stage3 interpolation. Other
labels remain baseline. Union cache knots with the baseline complete topology.
T55 independently reviewed chronology, rank flags and strict tie ownership.

## Frozen alternatives and falsifiers
1. Label2's +.00305512 baseline error comes primarily from final interpolation:
   its component absolute error should decrease materially under direct cache.
2. Earlier errors dominate or cancel: component/total can worsen independently.
   Report both; never replace the challenge metric or truth-select combinations.
3. Boundary or chronology errors: compare every primitive breakpoint and fixed
   u values to direct scalar evaluation after explicit target insertion, plus
   constant/duplicate/Wh0 and1e-16 scaling controls.
4. Outer rule changes masquerade as gains: baseline/hybrid use identical union
   q-panels, GL2/4 refinement below1e-7 and10% of any claimed improvement.

Controls and h9 topology count first, cap2000 panels. Then exactly four runs:
baseline/hybrid x GL2/4 on H2T4 seed0, same h9 and all later grids. No direct
label0 combination, noise/grid/rule sweep or random model inputs. Expansion
requires stable total error and error-squared CPU improvement, separately
reported component diagnostic, independent review and another frozen scope.

## Status
Claim5dbffa3 pushed; protocol frozen before implementation/measurements.

## Results
Pending. Full and per-label cone references are diagnostic only, never
estimator inputs. Reference total .30773074302392744.

## Compute accounting
Charge cache gate masks, original planes, intersection/sort/envelope work,
all cache CDF queries, prior grids/operators, union topology and target rules.
Clear fallback GL12 rule per run. Full CPU includes setup and runtime; named
operation counts remain partial/unaudited, no portable total or sampling claim.
One thread,2GB,<=110CPU cycles and project locks. No Monte Carlo/training.

## Claims
[inferred] Exact scalar terminal caching removes one interpolation for the
penultimate label relative to unchanged prior grids.
[speculative] This helps the total accuracy/cost criterion.

## Alternatives not ruled out
Other labels and earlier grid errors, conditional boundary representations,
and fortuitous cancellation. Results on one base do not settle noisy models.

## Failure modes
u-only boundary ownership with positive conditional mass, first-index ties,
missing line intersections, floating degeneracy/narrow CDF cancellation, and
mistaking complete real-arithmetic topology for a full floating error bound.

## Primitive checkpoint
`controls.py` passes in .548CPU under the review lock. Direct scalar evaluation
after explicit target insertion agrees with cache values at every discovered
u breakpoint and fixed probes within9.87e-15 for trained/zero/duplicate/Wh0
cases. Positive scale1e-16 discrepancy is5.56e-17; full constant/duplicate
retained-mass checks pass. At h9,664 cached pieces and398 knots expand1190
baseline panels to1532, within the frozen2000 cap. Cache setup .076CPU and
47,008 bytes; full plane/intersection/envelope counts in controls.json.
