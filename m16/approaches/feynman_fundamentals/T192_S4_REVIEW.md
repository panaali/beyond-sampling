# T192 S4 and noisy-copy evidence review

Reviewed branch snapshot `2930bc92` (the S4 source/results/README are unchanged from `a5ab8757`). Saved checker: `check_t192_s4_saved.py`; complete record and byte pins: `t192_s4_saved_review.json`. No author imports, model evaluations, geometry enumeration, mass integration or Monte Carlo were executed.

The four saved prefix summaries reproduce **34, 792, 21,032, 632,394** gate-pattern counts, and each declares all its subtrees complete. Growth arithmetic is **23.2941, 26.5556, 30.0682**. The summaries do not contain per-subtree gate sets, feasible interior witnesses, or exact geometry certificates, so this review cannot independently establish their cardinalities. T219's independent recount remains necessary. `prefix.py` accumulates selected gate words directly, while the noisy full-enumeration driver accumulates network patterns at rounded leaf centroids. Both use floating geometry with nonzero tolerances and rank thresholds.

The claimed T=10 projection **15,367,174,200,000** follows if every one of the five unseen steps multiplies the k=5 count by30. The three measured ratios do not imply that future lower bound. The README itself retains slowing growth after k=5 as an alternative. The DRI may make a practical no-go decision for the existing enumeration route, but the extrapolation is conditional, not a proved complexity lower bound or exclusion of aggregation methods.

`prefix.py` labels front elapsed wall time plus summed worker `time.time()` durations as `sec_cpu`. These are not process CPU measurements. Its partial-run claim of unbiased `cells_est` also does not follow from shuffled scheduling: wall-time stopping and unordered completion favor faster subtrees. For example, two simultaneous subtrees with runtimes1/100 and sizes1/1000, stopped on the first result past a0.5 deadline, yield an estimate2 instead of1001. The complete k2..5 summaries do not use this partial estimator.

All **18 noisy/clean rows and432 per-half records** reconcile saved accuracy, leaf sums and additive counters. The exact extrema across the16 noisy copies are:

| Quantity relative to clean | Minimum | Maximum |
|---|---:|---:|
| cells |11239/12179|5534/4875|
| reported regions |12599/13371|179/148|
| reported operation proxy |691383230/734656237|4227606267/3167681372|

Thus the rounded reported ranges reproduce the source table. The region totals are present only in summaries; the per-half logs omit pattern sets. The experiment covers two copies per noise level of **H8T4 and H16T4 zoo0**, with shared streams across relative noise scales. It does not establish a bound for arbitrary noisy copies or the target M16T10. Numerical accuracies and inherited operation proxies retain earlier T192 qualifications.

The older missing-array inventory remains accurate at its pinned `a5ab8757`. Commit `de4b8350`, now contained in `2930bc92`, adds `H8T5_zoo0.npz` and `H8T5_zoo2.npz` (1,770,680 and1,704,578 bytes). This review only hashes those blobs. T215's schema, model/environment and conditional `SLIVER` qualifications still apply; their arrival does not authorize a numerical extension contrary to current DRI priorities.

Scientific acceptance remains with the DRI. This is saved arithmetic/source review, not an independent target-region recount, complete FLOP audit, or new numerical reference.
