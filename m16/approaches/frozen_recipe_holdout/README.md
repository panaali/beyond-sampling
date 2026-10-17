# Fixed held-out H8,T4 seed1 (T148)

## Question
Does the numerical repair frozen on seed0 remain accurate on the next checkpoint without changing a parameter, and is the captured recipe approximation still small?

## Method
Seed1 checkpoint `2nd_argmax_8_4_1073741824_1.pth`, original model structure and pinned T131 estT snapshot. Candidate fixed before this checkpoint: recipe108 eps=.1, wideGL24/power2, thinGL96/power3, maxpair threshold.95, determinant cutoff1e-12, all other code unchanged. No parameters chosen from seed1 results. Capture exact and recipe108 output-error geometries separately with mass recording only. Preserve alloff first-index controls, coefficient-identical readout audit, original callbacks and hashes. Approximate cells need not agree with the true network pointwise.

For each mode replay original and candidate mass plus independent positive radial integration on exactly the same captured rays. Select positive order20/32/48 by conditional per-cone bound1e-9/N, not by an observed discrepancy. Eight fixed12-parent batches per mode, each≤55CPU/2GB/one thread, primary lock. Capture≤55CPU/200M namedgeometry operations. Save each completed parent/batch and every failure/negative; no normalization/clamping. Analytic orthant/order controls before replay. Original values/counters must reproduce existing final3 seed1 before interpretation. Compare recipe numerical bias, exact numerical bias, positive recipe-minus-exact approximation, and candidate total discrepancy. Report per-cone absolute errors and negatives as well as signed totals.

Alternative1: seed0 finite-rule repair generalizes (fixed holdout is a falsifiable test). Alternative2: other trained geometry defeats the same order (per-cone and aggregate errors against independent positive reference). Alternative3: recipe gate approximation becomes limiting (positive recipe versus positive exact). Failed reproduction or ties remain separate implementation concerns. No new MC, weight training, noise tuning, sharedsource changes or whole-rung claim.

## Status
Bounded held-out test complete; independent aggregate review e7b4aee passes. Claim d29d27b and protocol0572442 preceded implementation. T145 source/array review and T146 completecost audit run independently; no silent adoption of subsequent primitives.

## Results
Results below. Reproduce capture.py exact, capture.py recipe108; replay.py MODE BATCH for each mode and BATCH0..7, then summarize.py. Same project interpreter, one-thread variables,2GB and locks as earlier tasks.

## Compute accounting
All capture, rule setup, covariance/radial work, exact checks, prior-reference dependencies and failed passes count. Saved geometry is a validation convenience, not free production computation. Inherited named counters remain partial; T146 examines fullcost. Different CPU/counter categories are reported separately. No sampling advantage inferred solely from a partial counter or a single checkpoint.

## Claims
[inferred] A frozen held-out checkpoint tests generalization of the repaired numerical rule while separating gate approximation. [speculative] It may support a subsequent full existing panel; T149 remains blocked until independent holdout/cost handoffs.

## Alternatives not ruled out
Higher/other orders, positive representations, stable endpoints, better pruning certificates, coarser families and more dimensions. No finite result settles these; no rule selected on this holdout.

## Failure modes
Thin geometry, covariance cancellation, approximate gate failures, output ties, determinant cutoff, imperfect positive bounds, uncertain portablecost and ambiguous floatinggeometry. Preserve partial results and bounds that exclude arithmetic/geometry uncertainty. Exactmode here names the source gate-enumeration setting, not a proof of exact real-network accuracy.

## Held-out result
Both modes complete all96 parents. Exact capture has8447 leaves/30726 error simplices, recipe4573/19746, matching historical cell counts. All-off and coefficient-identical active readout counts are zero. Source original values and inherited full counters reproduce final3 seed1 exactly. This checkpoint was held out from evaluation of the new fixed candidate; existing historical final3 aggregate source/MC summaries were known, so this is not a blinded historical-data study.

| Mode | Original accuracy | Doubled-rule accuracy | Independent positive accuracy |
|---|---:|---:|---:|
| Exact gate enumeration | .9862761800428719 | .9874382753767020 | .9874600108260589 |
| recipe108 eps=.1 | .9865958970864601 | .9874384324269055 | .9874543286232546 |

The positive recipe-minus-exact approximation difference is **−5.68220280439e-6**, with combined conditional integration bound1.70946893939e-10. The candidate recipe's total difference is **−2.15783991534e-5**, comprising that approximation term and numerical term−1.58961963490e-5. The original recipe difference was−.000864113739599. Thus this unchanged repair improves total discrepancy about40x, while the remaining discrepancy is larger than seed0's+4.76456e-6. No setting was changed in response.

Per-cone recipe absolute numerical-error sum drops .00134494359485→.0000514632127154;562 negative candidate terms remain, signed sum−4.10432426866e-6. Exact mode retains1420 negative candidate terms and absolute-error sum.0000741975620548. No values are clamped. Positive-rule bounds are7.90611880321e-11 exact and9.18857059069e-11 recipe, no target misses; both exclude floating geometry and arithmetic. Corrected exact accuracy is.310 standard errors above existing4M sampling .98744275, a diagnostic with sampling uncertainty.

Capture costs14.290490286/11.230251069CPU exact/recipe; three-way replay costs156.611059614/103.637036137CPU in bounded batches. Original full partial counters207257255/147225326 reproduce; doubled counters376677671/262248590 include their source geometry counters but remain incomplete portable costs. Controls separately total92544 named units per mode. T146/T153 native expression trace and T150 opaque-kernel review are still needed before a full-cost conclusion.

Independent T55 e7b4aee checks all16 complete arrays, exhaustive indices, source hashes, exact sums of saved binary64 values, both counter identities and decomposition. Independent T151432ad13/add47fb verifies all30726 exact-region fixed interior centers and64 selected exact integer/Fraction recurrences against the direct network; these are point checks, not whole-cone/completeness proofs. No direct true-network membership is assumed for approximate recipe cells.

[measured] The frozen numerical repair transfers to this second checkpoint with a substantial improvement, but weaker residual accuracy than on its development checkpoint. [inferred] This supports broader fixed-panel testing once cost dependencies are satisfied. It does not establish a full model/noise panel, a general bound, a passed rung or the H16,T10 challenge.
