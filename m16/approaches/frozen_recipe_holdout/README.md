# Fixed held-out H8,T4 seed1 (T148)

## Question
Does the numerical repair frozen on seed0 remain accurate on the next checkpoint without changing a parameter, and is the captured recipe approximation still small?

## Method
Seed1 checkpoint `2nd_argmax_8_4_1073741824_1.pth`, original model structure and pinned T131 estT snapshot. Candidate fixed before this checkpoint: recipe108 eps=.1, wideGL24/power2, thinGL96/power3, maxpair threshold.95, determinant cutoff1e-12, all other code unchanged. No parameters chosen from seed1 results. Capture exact and recipe108 output-error geometries separately with mass recording only. Preserve alloff first-index controls, coefficient-identical readout audit, original callbacks and hashes. Approximate cells need not agree with the true network pointwise.

For each mode replay original and candidate mass plus independent positive radial integration on exactly the same captured rays. Select positive order20/32/48 by conditional per-cone bound1e-9/N, not by an observed discrepancy. Eight fixed12-parent batches per mode, each≤55CPU/2GB/one thread, primary lock. Capture≤55CPU/200M namedgeometry operations. Save each completed parent/batch and every failure/negative; no normalization/clamping. Analytic orthant/order controls before replay. Original values/counters must reproduce existing final3 seed1 before interpretation. Compare recipe numerical bias, exact numerical bias, positive recipe-minus-exact approximation, and candidate total discrepancy. Report per-cone absolute errors and negatives as well as signed totals.

Alternative1: seed0 finite-rule repair generalizes (fixed holdout is a falsifiable test). Alternative2: other trained geometry defeats the same order (per-cone and aggregate errors against independent positive reference). Alternative3: recipe gate approximation becomes limiting (positive recipe versus positive exact). Failed reproduction or ties remain separate implementation concerns. No new MC, weight training, noise tuning, sharedsource changes or whole-rung claim.

## Status
Claim d29d27b pushed before implementation. Protocol frozen. T145 source/array review and T146 completecost audit run independently; no silent adoption of subsequent primitives.

## Results
Pending. Reproduce capture.py exact, capture.py recipe108; replay.py MODE BATCH for each mode and BATCH0..7, then summarize.py. Same project interpreter, one-thread variables,2GB and locks as earlier tasks.

## Compute accounting
All capture, rule setup, covariance/radial work, exact checks, prior-reference dependencies and failed passes count. Saved geometry is a validation convenience, not free production computation. Inherited named counters remain partial; T146 examines fullcost. Different CPU/counter categories are reported separately. No sampling advantage inferred solely from a partial counter or a single checkpoint.

## Claims
[inferred] A frozen held-out checkpoint tests generalization of the repaired numerical rule while separating gate approximation. [speculative] It may support a subsequent full existing panel; T149 remains blocked until independent holdout/cost handoffs.

## Alternatives not ruled out
Higher/other orders, positive representations, stable endpoints, better pruning certificates, coarser families and more dimensions. No finite result settles these; no rule selected on this holdout.

## Failure modes
Thin geometry, covariance cancellation, approximate gate failures, output ties, determinant cutoff, imperfect positive bounds, uncertain portablecost and ambiguous floatinggeometry. Preserve partial results and bounds that exclude arithmetic/geometry uncertainty. Exactmode here names the source gate-enumeration setting, not a proof of exact real-network accuracy.
