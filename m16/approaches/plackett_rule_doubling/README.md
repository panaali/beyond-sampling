# Fixed uniform Plackett rule doubling (T140)

## Question
Does one fixed uniform quadrature-order doubling remove most of the demonstrated T131 integration bias at modest incremental named cost? This is a bounded numerical rescue, not a full estimator/rung evaluation.

## Method
Pinned T131 H8,T4 seed0 output-error archive has35098 simplices and unchanged source/independent positive values. Keep every source operation and branch choice, including max correlation>.95 selection, wide power2/thin power3 transforms, clamps and determinant cutoff. Change only wide GL12→24 and thin GL48→96 for all cones. One candidate is frozen before its evaluation; no parameter fitting, later selection or source edits. Use eight fixed12-parent batches, each≤55CPU/2GB/one thread/primary compute lock. Preserve partial checkpoints and all exceptions.

Analytic controls: coordinate orthant mass1/16 and coordinate-order half-orthant1/32; both source and doubled rules must pass. Compare signed total, absolute per-cone error sum/max/quantiles, negative counts and cutoff effects against saved independent positive reference. Its uncertainty remains conditional on captured floating geometry/arithmetic. Aggregate agreement can hide cancellation, hence per-cone checks are required. Review dependencies T131/T55 and T137 high-precision cause review remain separate.

Alternative1: insufficient fixed quadrature order causes bias; uniform doubling is a direct falsifiable rescue. Alternative2: covariance/inverse roundoff dominates; T137 same-expression MP reevaluation separates that case. Alternative3: unresolved endpoint peaks still defeat doubled finite rule; preserved per-cone errors and negatives expose this without declaring Plackett integration impossible. No adaptive-order or cheap certified error bound is tested here.

## Status
Bounded fixed-candidate experiment complete; independent review ea53df3 passes. Claim d56bea3 and protocol ceef8ba preceded implementation.

## Results
Measured results below. Reproduction: run.py BATCH for BATCH0..7, then summarize.py. No random model inputs or new MC.

## Compute accounting
Charge rule creation, import/setup, inversion and every inherited node operation, captured-source dependence and all failed passes. Report candidate named mass counter and delta from same captured old source counter204957048. Carry prior full source total240607902 only to show the same-counter hypothetical geometry+mass total; that inherited counter is partial and not a portable FLOP proof. Positive validation's81.39114827CPU plus14.991144678CPU capture remains prior reference cost, never a free estimator. Candidate CPU includes setup/checkpointing. No sampling-win claim from this one checkpoint or a partial counter.

## Claims
[inferred] A fixed order increase can discriminate finite integration bias from persistent failures without changing neural geometry. [speculative] It may provide a cheaper accurate mass primitive; this remains untested until evaluation and independent review.

## Alternatives not ruled out
Other fixed orders, determinant/eigenvalue endpoint diagnostics, adaptive rules, stable conditional-correlation formulas, analytic cancellation removal, positive cone representations, improved topology and alternate partitions. No finite negative exhausts these.

## Failure modes
Endpoint concentration, cancellation, near-singular covariance, clipped correlations, determinant cutoff, parameter-selection overfit, reference conditional bounds and incomplete portable cost. Preserve every domain error, nonfinite value, negative output and incomplete batch. No normalization or probability clipping.

## Bounded result
All35098 fixed simplices complete. Candidate accuracy **.9761652290904052** versus independent reference **.9761608075466027**; error-mass bias is−4.42154380245e-6, compared with old+1.11347160674e-3 (absolute total bias improves about252x). Per-cone absolute-error sum drops .00221562360584→.0000555451350264 (about39.9x); maximum per-cone error drops3.91415e-6→1.74488e-7. Thus improvement is not solely signed cancellation, but cancellation still reduces the total substantially. There remain735 negative candidate masses summing−6.78627790418e-6; none is clamped. No claim of nonnegative or certified probability output is made.

The73 source determinant-cutoff cones have summed positive reference mass2.188531825e-13, negligible for this panel but not a universal cutoff theorem. Reference total conditional integration bound1.04221660744e-10 excludes geometry/roundoff. T137 independent high-precision fixture review isolates finite quadrature bias in the original wide12/power2 branch; maxpair correlation alone misses that case.

Candidate named mass count **404438808** equals2×204957048−156×35098:156 setup units per cone and152 per evaluated node, with unchanged branch/cutoff. The same-counter hypothetical full geometry+mass total is **440089662**, versus240607902 original. This inherited partial count omits kernel/setup categories and is not a portable FLOP certificate. Candidate replay CPU97.669149593 includes controls, rule generation, provenance, extra branch-diagnostic inversions and saving; controls contribute92544 named units separately. Source/positive reference costs remain prior dependence. One process launch failed before execution because the exec transport disconnected; retry succeeded, no scientific data was lost. Partial batch4 checkpoint was committed during execution and then replaced by its complete result; final summary requires complete disjoint indices.

[measured] One fixed order doubling sharply reduces both signed and absolute numerical bias on this checkpoint at roughly double the inherited mass-rule counter. [inferred] This rescues a plausible inexpensive quadrature variant and prevents interpreting the original finite-rule failure as a negative for all Plackett representations. It does not establish the remaining recipe approximation error, all-seed/noise performance, portable cost, H16T10 scalability, or a passed rung. Independent T55 saved-array/source/counter review requested.

Independent T55 review ea53df3 verifies all35098 saved indices, raw sums/absolute errors/negative terms, all source-style branch flags and named-cost identities in.587CPU. Nearest rmax to.95 is5.49e-6, so diagnostic branch arithmetic agrees with the actual source on this corpus. T140 is handed off; T145 separates unchanged recipe pruning error, while T146 independently audits full sourcecost. No additional rule candidate is selected here.
