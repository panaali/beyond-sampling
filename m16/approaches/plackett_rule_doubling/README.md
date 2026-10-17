# Fixed uniform Plackett rule doubling (T140)

## Question
Does one fixed uniform quadrature-order doubling remove most of the demonstrated T131 integration bias at modest incremental named cost? This is a bounded numerical rescue, not a full estimator/rung evaluation.

## Method
Pinned T131 H8,T4 seed0 output-error archive has35098 simplices and unchanged source/independent positive values. Keep every source operation and branch choice, including max correlation>.95 selection, wide power2/thin power3 transforms, clamps and determinant cutoff. Change only wide GL12→24 and thin GL48→96 for all cones. One candidate is frozen before its evaluation; no parameter fitting, later selection or source edits. Use eight fixed12-parent batches, each≤55CPU/2GB/one thread/primary compute lock. Preserve partial checkpoints and all exceptions.

Analytic controls: coordinate orthant mass1/16 and coordinate-order half-orthant1/32; both source and doubled rules must pass. Compare signed total, absolute per-cone error sum/max/quantiles, negative counts and cutoff effects against saved independent positive reference. Its uncertainty remains conditional on captured floating geometry/arithmetic. Aggregate agreement can hide cancellation, hence per-cone checks are required. Review dependencies T131/T55 and T137 high-precision cause review remain separate.

Alternative1: insufficient fixed quadrature order causes bias; uniform doubling is a direct falsifiable rescue. Alternative2: covariance/inverse roundoff dominates; T137 same-expression MP reevaluation separates that case. Alternative3: unresolved endpoint peaks still defeat doubled finite rule; preserved per-cone errors and negatives expose this without declaring Plackett integration impossible. No adaptive-order or cheap certified error bound is tested here.

## Status
Claim d56bea3 pushed before work. Protocol frozen; implementation pending.

## Results
Pending. Reproduction: run.py BATCH for BATCH0..7, then summarize.py. No random model inputs or new MC.

## Compute accounting
Charge rule creation, import/setup, inversion and every inherited node operation, captured-source dependence and all failed passes. Report candidate named mass counter and delta from same captured old source counter204957048. Carry prior full source total240607902 only to show the same-counter hypothetical geometry+mass total; that inherited counter is partial and not a portable FLOP proof. Positive validation's81.39114827CPU plus14.991144678CPU capture remains prior reference cost, never a free estimator. Candidate CPU includes setup/checkpointing. No sampling-win claim from this one checkpoint or a partial counter.

## Claims
[inferred] A fixed order increase can discriminate finite integration bias from persistent failures without changing neural geometry. [speculative] It may provide a cheaper accurate mass primitive; this remains untested until evaluation and independent review.

## Alternatives not ruled out
Other fixed orders, determinant/eigenvalue endpoint diagnostics, adaptive rules, stable conditional-correlation formulas, analytic cancellation removal, positive cone representations, improved topology and alternate partitions. No finite negative exhausts these.

## Failure modes
Endpoint concentration, cancellation, near-singular covariance, clipped correlations, determinant cutoff, parameter-selection overfit, reference conditional bounds and incomplete portable cost. Preserve every domain error, nonfinite value, negative output and incomplete batch. No normalization or probability clipping.
