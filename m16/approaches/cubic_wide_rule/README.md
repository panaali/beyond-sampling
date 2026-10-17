# Fixed cubic wide-rule discriminator (T154)

## Question
Can a fixed endpoint transform change reduce the original wide-rule error without increasing its number of nodes? T137 verified finite12/power2 quadrature error on a negative fixture; T142 found a small covariance eigenvalue despite moderate pair correlations.

## Method
Exactly one candidate: change wideGL12/power2 to wideGL12/power3; retain thinGL48/power3, maxpair threshold.95, determinant cutoff and every source operation. Use only the previously diagnosed seed0 exact error archive35098rays, with existing original, T140 doubled24/96 and independent positive values. This is exploratory on a known corpus, not a new heldout test. It cannot replace the frozen24/96 full-panel candidate T149 by selecting the better aggregate result.

Analytic controls before trained measurements: orthant1/16, coordinate-order halforthant1/32, and independent120-degree dual-normal pair blocks giving mass1/36. Both source and candidate must agree within1e-12. Run eight fixed12-parent batches≤55CPU/2GB/one thread/primary lock; retain every negative, partial result, exception and source provenance. Compare signed totals and per-cone absolute-error sum/max/quantiles with positive references, retaining cancellation and cutoff contributions. No clamping, tuning, new geometry, new model input probes, MC or seed1 evaluation.

Alternative1: inadequate endpoint clustering drives wide-rule error (fixed cubic transform tests this cheaply). Alternative2: fewer interior nodes harm some cones even if the total improves (absolute-error distributions and negatives expose this). Alternative3: covariance arithmetic dominates (T137 fixed-fixture MP audit found it negligible only for that case; not a universal exclusion). No broad negative from this one transform.

## Status
Existing Ready T154 claimed and pushed84f767a after T148 independent handoff. Protocol frozen; all eight batches complete. Independent T55 saved-array review requested.

## Results
All 35,098 indices were evaluated once. Accuracy is .9761281369336525, versus positive reference .9761608075466027 and doubled-rule .9761652290904052. The cubic signed error mass is +3.2670612950133484e-5 and per-cone absolute-error sum .00023151691021391528, improvements of about 34.1x and 9.57x over the original rule. The doubled rule remains better on both measures. Cubic leaves 2,286 negative contributions, totaling −.00005803035579941216; none were clipped. Maximum cone error is 4.1991142276240464e-7. Thin/cutoff values are bitwise unchanged. Reference integration-only bound is 1.0422166074433452e-10 and excludes geometry and floating roundoff.

Reproduce with `/workspace/clean-neurons/.venv/bin/python m16/approaches/cubic_wide_rule/run.py BATCH` for BATCH 0 through 7, then the sibling `summarize.py`, with one-thread resource limits and the primary compute lock. Raw batches and summary retain full error distributions and controls. The original summary compilation failed on missing whitespace in two conditional expressions; `summary_syntax_failure.json` preserves it. Only token spacing was repaired before aggregation; numerical artifacts were unaffected.

## Compute accounting
Same source branch/cutoff and node counts imply same inherited mass counter204957048 on this archive. Rule-generation powers/Jacobians differ and their setup/CPU are charged; equal node count does not imply identical hardware time or a complete portable cost. Source/positive references and T140 costs remain prior dependencies. Candidate recorded CPU is 52.023945467 seconds, including analytic controls charged 95,040 named units. The inherited hypothetical full counter remains 240,607,902. The failed compilation adds unmeasured CPU (observed wall time .035839375 seconds). T146/T150/T153 cost audits remain independent.

## Claims
[measured] On this known corpus the fixed cubic transform reduces original signed and absolute error at unchanged node count. [inferred] Endpoint placement matters, but this variant leaves substantial residual error and is less accurate than the frozen doubled rule. [speculative] Other justified transforms may offer a better cost/accuracy tradeoff; no model-panel generalization or selection claim.

## Alternatives not ruled out
Other fixed transforms, justified endpoint splits, eigenvalue-based rules, adaptive/error-certified integration, positive representations and geometry changes. No automatic additional sweep follows this candidate.

## Failure modes
Endpoint/interior tradeoff, cancellation, negative mass, inverse conditioning, determinant cutoff, reference geometry/roundoff limits and exploratory-corpus selection. Preserve all errors and unresolved alternatives.
