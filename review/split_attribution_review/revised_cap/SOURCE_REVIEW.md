# Frozen source findings

## What changed

At `876f9ddd`, `cap_bound` replaces `sqrt(1-cos²)` with the norm of the residual `r-(r·u)u`, adds `SIN_PAD=1e-14`, applies `BETA_REL=1e-9`, and falls back to one when the computed minimum cosine is at most `1e-6`. This avoids the original narrow-cone cancellation. The ideal argument is sound: a circular cone of angle below pi/2 that contains every generator contains their positive hull. Its Gaussian probability is one half the regularized beta function at sin²(theta).

However, unit rays and normalized axis are mathematical premises; their stored values arise from floating normalization. The source provides no error derivation covering the sum, its norm, normalization, dot products, residual norms, squared argument and SciPy beta evaluation over a declared finite domain. No cited version-specific beta error theorem binds the relative padding. Fixed padding can be useful engineering evidence without being a proof. This review does not assert a pointed trained counterexample to the new pad.

## Zero-axis/domain qualification

For the exact unit rows `+e_i,-e_i` in dimension three, the positive hull is all of R³ and has Gaussian mass one. The row sum is exactly zero. Under ordinary NumPy nontrapping invalid arithmetic, normalization produces NaNs; the comparison `cs.min() <= 1e-6` is false. Python `min(1.0, nan)` keeps its first operand, so `sn=1`, beta at one equals one, and the return expression is approximately `0.5000000005`, below one. This is a source-semantics argument, not an executed fixture.

The inherited `Cone` documentation expressly describes **pointed** cones, so the example is outside that caller contract. It refutes an unrestricted positive-hull/API interpretation, not the trained pointed traversal. It nevertheless shows that an undefined axis is not explicitly turned into the advertised safe fallback. The author's antipodal controls use rows `u,u,-u` (with further copies of u in larger dimensions), whose sum is nonzero; they do not test this case. A finite/nonzero-axis guard would address the domain edge without proving the remaining rounding bounds.

## What the 834 checks retain

`capcheck.py` specifies 225 narrow random cases (three dimensions, 15 scales, five per scale), 90 repeated/antipodal cases, and a variable number of clipped sub-cones. The final text reports 834 checks and zero violations, implying 519 sub-cone comparisons if execution followed that source. The text retains only the first narrow case at each scale unless a violation occurs, plus repeated/antipodal summaries. It does not retain every ray matrix, every reference, source/runtime hashes or exception count. A NaN reference is explicitly not counted as a violation. Consequently an independent all-834 numerical reconstruction is not possible from the printed text alone without rerunning the author work, which this phase does not authorize.

For d3, the solid-angle reference is a distinct formula, still floating arithmetic. For d4/d5 and the recipe sub-cones, references use `estT.orthant`/`estT.mass`, not certified exact integrals. `capcheck.py` imports `estT` before importing `prune`; its explicit path insertion supplies the split folder and arc reproduction folder, neither of which contains a tracked estT module at this commit. The effective external import path/runtime therefore needs provenance before binding the text to a particular estT implementation. This is not proof that the author imported the wrong module.

## Pruning budget scope

`MASS_PAD=1e-15` is added per pruned cell only in the mass-test branch. Its comment motivates it from typical d3 noise, without a universal bound or a d4/d5 argument. The threshold still tests the unpadded `mb`, while the accumulated bound uses the pad; this is not by itself inconsistent because the budget is recorded afterward. `by_step` stores the unpadded mass, so a saved checker must account for the difference rather than demand exact equality with the padded global sum.

In the cap branch, a wrong-centroid cell still contributes numerical `estT.mass` to the estimate. Even a valid cap bounds ideal classification replacement only under correct domain coverage. It does not automatically bound error in those numerical masses or the unpruned recipe, and ordinary floating addition of `P['bound']` is not outward accumulation. The report docstring now correctly says its B ratio is pruning-only and does not certify recipe108's error. The broader source expressions calling masses exact/certified remain too strong without those premises.

## Replay and cost provenance

The pinned tree has 50 nonzero-threshold T3 rows plus one zero-threshold T3 seed0 row. The previous T4 row is deleted, and the new tree supplies no T4/T5 pruning result. The shell queue is intent, not coverage. Any cross-T growth claim therefore remains unavailable here.

Commit `2f097739` changes report comparison from each result's historical `N` to `flops / S_T`, with S_T=511/682/853. The JSON `N` fields remain the old `fpe_flops` convention; they must not be silently interpreted as canonical draws. `flops=2*ops` remains the declared counter convention, not a newly audited complete native cost. The report uses continuous draw budgets rather than integer floor budgets and point-valued reference accuracies. These are explicit comparison conventions, not confidence intervals or a full rung decision. There is no rendered report file for text parity at this pin.
