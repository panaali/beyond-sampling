# Fixed cubic wide-rule discriminator (T154)

## Question
Can a fixed endpoint transform change reduce the original wide-rule error without increasing its number of nodes? T137 verified finite12/power2 quadrature error on a negative fixture; T142 found a small covariance eigenvalue despite moderate pair correlations.

## Method
Exactly one candidate: change wideGL12/power2 to wideGL12/power3; retain thinGL48/power3, maxpair threshold.95, determinant cutoff and every source operation. Use only the previously diagnosed seed0 exact error archive35098rays, with existing original, T140 doubled24/96 and independent positive values. This is exploratory on a known corpus, not a new heldout test. It cannot replace the frozen24/96 full-panel candidate T149 by selecting the better aggregate result.

Analytic controls before trained measurements: orthant1/16, coordinate-order halforthant1/32, and independent120-degree dual-normal pair blocks giving mass1/36. Both source and candidate must agree within1e-12. Run eight fixed12-parent batches≤55CPU/2GB/one thread/primary lock; retain every negative, partial result, exception and source provenance. Compare signed totals and per-cone absolute-error sum/max/quantiles with positive references, retaining cancellation and cutoff contributions. No clamping, tuning, new geometry, new model input probes, MC or seed1 evaluation.

Alternative1: inadequate endpoint clustering drives wide-rule error (fixed cubic transform tests this cheaply). Alternative2: fewer interior nodes harm some cones even if the total improves (absolute-error distributions and negatives expose this). Alternative3: covariance arithmetic dominates (T137 fixed-fixture MP audit found it negligible only for that case; not a universal exclusion). No broad negative from this one transform.

## Status
Existing Ready T154 claimed and pushed84f767a after T148 independent handoff. Protocol frozen.

## Results
Pending. Reproduce run.py BATCH for0..7, then summarize.py with project interpreter and one-thread resource limits.

## Compute accounting
Same source branch/cutoff and node counts imply same inherited mass counter204957048 on this archive. Rule-generation powers/Jacobians differ and their setup/CPU are charged; equal node count does not imply identical hardware time or a complete portable cost. Source/positive references and T140 costs remain prior dependencies. Record candidate CPU, analytic-control counter and all setup/failure work. T146/T150/T153 cost audits remain independent.

## Claims
[inferred] A fixed transform can discriminate endpoint resolution from order alone without changing neural geometry. [speculative] It may be a useful lower-node alternative; no model-panel generalization or selection claim.

## Alternatives not ruled out
Other fixed transforms, justified endpoint splits, eigenvalue-based rules, adaptive/error-certified integration, positive representations and geometry changes. No automatic additional sweep follows this candidate.

## Failure modes
Endpoint/interior tradeoff, cancellation, negative mass, inverse conditioning, determinant cutoff, reference geometry/roundoff limits and exploratory-corpus selection. Preserve all errors and unresolved alternatives.
