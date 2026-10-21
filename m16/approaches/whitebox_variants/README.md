# T476: bounded deadline variants

## Question
Can a small, preregistered set of V25 variants improve deadline submission cost or reliability?

## Method
Use the previously verified T474 row-blocked V25 and its four pinned public reference rows. Freeze all environment knobs in each standalone estimator. First test 256-row blocking on mini_0 and mini_1, compare predictions to baseline. Then test three independent changes: old ranks 320/192 (moderate reduction; upstream warns 256/128 can cliff); old-source age 3 rather than 4 at original ranks; enable the original 13-feature mean correction. Screen initially planned mini rows then at most two winners on full rows; execution instead scored all four variants on all four existing rows. Treat every row as DEV, with no held-out-selection claim. No more than 480 worker CPU seconds, one BLAS thread, 1.65 GB RSS supervisor guard.

Alternative explanations: lower rank may save FLOPs but destroy accuracy; paired raw-MSE/cost measurements separate these. Larger blocks may reduce Python residual overhead while changing rounding or exceeding memory; baseline parity and RSS distinguish these. Old fitted correction may be stale; public row improvements are not evidence of fresh-model generalization.

## Status
V25 screen complete; block256 is bitwise-identical on all four rows and has more residual-time headroom. Rank320, age3 and old correction variants lose adjusted error. These negatives are unverified and scoped to this panel. Stronger e30 follow-up now authorized by root.

## Results
Machine-readable results and exact commands in this folder; registry pending root review. Public rows are development data, not sealed evaluation.

## Compute accounting
All prediction operations metered using flopscope 0.12.3. Setup separately metered. Sum worker CPU in summary; no training/reference-generation compute. Baseline and references reused from T474.

## Claims
[speculative] Moderate rank reduction may improve adjusted error. [speculative] Larger blocks may improve residual time. No claim is confirmed.

## Alternatives not ruled out
Strassen, dead-neuron pruning, newer fitted corrections and fixed-basis transport are stronger public techniques assigned separately to T475. Exhaustive rank/age tuning is not attempted under this deadline.

## Failure modes
Public-panel overfitting; stale correction tables; residual wall-clock variance; numerical differences from blocking; off-shape source instability inherited from V25.
