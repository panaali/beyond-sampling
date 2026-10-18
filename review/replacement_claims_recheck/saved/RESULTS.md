# Sole retained-summary check passed; independent saved audit pending

Source375055b5/0434b76c/bedce167 was cleared by h4 review44dd9165 (shared7619d8a2). Execution scope00a5f981 and immutable gate79dd6158 were published through6707b0ac before the sole run. Raw ddfc9a23 retains63 checks,237 uniquely compatible Boolean counts and four progress events, with no failure/retry or unknown count. Runtime records20/20CPU seconds and2GiB/2GiB limits. Internal main-scope CPU was0.368447621; no historical or native cost inference.

For N=200000, conditional on the documented correctly rounded historical mean-division assumption, the five retained confusion proportions uniquely reconstruct:

| Outcome | Compatible count |
|---|---:|
| Both correct |186597|
| Network correct, replacement wrong |3897|
| Network wrong, replacement correct |4117|
| Both wrong, same label |4482|
| Both wrong, different labels |907|

Disagreement is8921/200000; agreement is.955395. Network accuracy is.95247 and replacement accuracy.95357. Task gain is220/200000=.0011 (0.11 percentage points). The exact paired sample-mean variance is4006879/19999900000000; this does not include training/family-selection uncertainty or independently validate model outputs. The near/blind union has compatible support39132 and7002 disagreements. All tested rank-case, gap-bin, gate-frequency near/far, error-conditional and picked-rank identities reconcile. These are reconstructed aggregate counts, not retained raw labels.

Validation family selection, evaluation-selected neuron ordering, all16 group coverage and existing final-step-only endpoint metrics also agree with the pinned source. Seven iid-noise rows and separate direct-fit rates are checked for integer compatibility only; their realized R-squared, error mechanism or feature ceiling is not established.

No raw x/ref/p_all archive or coefficient model was recovered. T258's independent-code/different-seed reproduction within0.5pp and D1 replication therefore remains OPEN, not failed or complete. Independent saved review by T55 is pending. New fitting must follow the separately frozen resource/provenance plan, and should retain coefficients for T260 rather than duplicate training. Prior source claims and all untried alternatives remain qualified.
