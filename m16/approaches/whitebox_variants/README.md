# T476: bounded deadline variants

## Question
Can a small, preregistered set of V25 variants improve deadline submission cost or reliability?

## Method
Use the previously verified T474 row-blocked V25 and its four pinned public reference rows. Freeze all environment knobs in each standalone estimator. First test 256-row blocking on mini_0 and mini_1, compare predictions to baseline. Then test three independent changes: old ranks 320/192 (moderate reduction; upstream warns 256/128 can cliff); old-source age 3 rather than 4 at original ranks; enable the original 13-feature mean correction. Screen initially planned mini rows then at most two winners on full rows; execution instead scored all four variants on all four existing rows. Treat every row as DEV, with no held-out-selection claim. No more than 480 worker CPU seconds, one BLAS thread, 1.65 GB RSS supervisor guard.

Alternative explanations: lower rank may save FLOPs but destroy accuracy; paired raw-MSE/cost measurements separate these. Larger blocks may reduce Python residual overhead while changing rounding or exceeding memory; baseline parity and RSS distinguish these. Old fitted correction may be stale; public row improvements are not evidence of fresh-model generalization.

## Status
V25 screen complete; block256 is bitwise-identical on all four rows and has more residual-time headroom. Rank320, age3 and old correction variants lose adjusted error. These negatives are unverified and scoped to this panel. Stronger e30 follow-up completed. All four e30 parameter variants were finite and passed local contract gates. None beats baseline mean adjusted error on the four-row DEV panel; rank384, rank448 and tau3.0 remain near-ties eligible for owner-authorized official comparison. Tau2.7 is materially worse and excluded. No further heavy runs remain.

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

## E30 extension frozen before execution
Root authorized stronger-public-baseline variants after the first screen. The owner waived local memory limits; supervisor now allows 7.5 GB, while the competition 8 GB still binds. Pin e30_a source SHA6ddd97a96c30397683815c530bfb422f0a49ffdea49e4f7827149ed0a14729b0 and its MIT license. Four variants: shared ranks384/192, rank448/224, dead threshold2.7, and dead threshold3.0. Each compares error/cost on the same four DEV rows. All changed settings are literals in submitted source; no reliance on environment overrides. Alternative explanation: a public best setting may already dominate these local changes, and random small-panel shifts may not transfer. The first rank384 run hit the too-tight 6 GB guard at 6.054 GB and was preserved; no numerical output was produced. The rerun uses 7.5 GB. No new-method claim. Cumulative deadline cap extended by root to12CPU-min; one worker/one thread.

## Reproduction

Use the T474 Python environment and source pins. Commands:

```sh
/tmp/whest-env-t473/bin/python m16/approaches/whitebox_variants/build.py
/tmp/whest-env-t473/bin/python m16/approaches/whitebox_variants/build_e30.py
/tmp/whest-env-t473/bin/python m16/approaches/whitebox_variants/screen.py block256 rank320 age3 corrected
/tmp/whest-env-t473/bin/python m16/approaches/whitebox_variants/screen.py e30_rank384 e30_rank448 e30_tau27 e30_tau30
/tmp/whest-env-t473/bin/python m16/approaches/whitebox_variants/summarize.py
```

The supervisor refuses to overwrite run records. Existing outputs are rescored without rerunning. `build_e30.py` requires the pinned public source in `/tmp/whitebox-review-e30`; `sources.json` records immutable hashes. Final source files are checked in and do not need regeneration for packaging. E30 derivatives retain `E30_LICENSE`; V25 derivatives retain `LICENSE`.

The rank448 source was checked to differ from public e30 only by its provenance header and the one designated literal. Rank384 changes precisely the two ranks; tau variants change precisely the threshold. Independent root/source review is requested before submission. `results/aggregate.json` independently rescored saved predictions and reports both required point bars against covariance, with no population interval. All successful candidates pass those bars; comparison against the much stronger e30 source is the relevant selection criterion.

The orchestration parent was paused between workers to let root repeat a residual-time check in isolation, then resumed on root instruction. No estimator worker was suspended or its metering changed.

## Independent review received
Codex `whitebox_stronger` independently checked the rank448 source difference (header plus the single rank literal), the block256 fixed-default/output-blocking rewrite, and bitwise equality of every saved block256 layer array against T474 on all four public models. Review was communicated to root before packaging. This supports the source transformation and saved-output parity only; it is not an independent fresh-model accuracy run or a confirmed performance claim.

## Frozen follow-up robustness check
After deadline candidates were frozen and submitted, root requested two additional public full rows (indices2/3), untouched by our current tuning. Test the frozen rank448 and tau3.0 sources on those rows with unchanged literals and hashes; stronger agent tests unchanged e30. No fitting or new settings. At most two additional CPU minutes, one worker, stop starting compute at23:57UTC. These are not sealed/private rows and upstream authors may have used them. References/provenance are archived by T475; results here under `fresh_scores.json`. Reproduce with `fresh_check.py <T475 fresh-reference-directory>`.

The independent reviewer subsequently also verified rank384 and tau3.0 as exactly their designated literal replacements plus provenance headers; hashes matched. Receipt: [T475 additional source review](../whitebox_stronger/results/t476_additional_source_review.json).

Fresh follow-up completed before the cutoff within its original CPU cap. All four frozen variant executions passed local gates. Rank448 slightly improves the mean over the two fresh rows; tau3.0 slightly worsens it. This is descriptive two-model evidence, with no confidence/generalization claim. Relative to frozen e30, rank448 passes both strict and lenient mean-score point bars; tau3.0 fails strict and passes lenient. Row-specific results, source hashes, costs and elapsed compute are preserved in `results/fresh_summary.json`. No model computation remains running and no source was changed after freezing.
