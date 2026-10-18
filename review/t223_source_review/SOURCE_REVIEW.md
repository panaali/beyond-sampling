# Independent source and proof review

The fixed source/findings reviewed are recover_910's ca3b6fb3 and the snapshot manifest of author 2809650d. Review does not import the author source.

## Exact statements and their domain
For independent symmetric continuous coordinates, all sign patterns have equal mass. At most one positive coordinate consists of T+1 patterns, hence (T+1)/2^T and 11/1024 for T=10. Conditional independent standard-Gaussian magnitudes are half-normal, independently of signs. Thus uniform selection of those orthants and independent half-normal magnitudes is an ideal map for this region. Finite scrambled Sobol nodes and clipped inverse CDFs are an approximation to that law; the weight alone says nothing about the network's conditional error.

Within the parent rank-2/3 chart, the second largest value is u+|t|/sqrt(2), giving the two exterior intervals when u<0; when u>=0 the omitted endpoint is null. Within rank-1/2 it is u-|t|/sqrt(2), giving the interior interval for u>0 and an empty nonblind interval for u<=0. The implementation restricts both signs and preserves the side's target index. These checks assume the parent's chart/domain decomposition. They do not independently establish the complete co-area normalization or exact floating integration. A face's sign u is not the entire fibre's blind membership.

## Numerical and cost scope
The parent face CDF uses a 160001-point grid on [-8,8], clipped inverse probabilities and tau<=8.5. The walker probes with a positive epsilon and stops after 2000 iterations without exporting an unfinished-fibre residual. It recomputes h/dh from zero through all T steps on every active cell, whereas its cost formula counts a restart suffix. The 2.23 correction and nominal 6000/5948 conversion are inherited conventions, not a full cost theorem. This is a source limitation, not an observed cap/truncation failure or measured bias in these saved rows.

The slope-zero output comparison replaces exact equality by a relative intercept tolerance against earlier labels. NumPy argmax itself owns exact ties by first index. Truth argsort specifies no first-index tie policy. Ideal continuous input ties are null, but that alone does not settle the finite node map.

The estimator uses stage-one outputs to allocate stage two and reuses those observations. Fixed seeds make repeated calls deterministic, but do not prove an unbiased adaptive randomized-rule estimator. The code also divides by the observed mean face cost; universal handling of zero observed face costs or requests exceeding preallocated nodes is not established. No such condition is asserted to have occurred in the published panel.

## Saved score algebra
For each raw row, the local convention charges cost C=raw cost times 2.23 for sci_tao, and C=raw cost for tie_blind, which already charges its factor internally. The grouped statistic is mean[(estimate-p)^2-se^2] / mean[p(1-p)/C]. This agrees with leaderboard.table; mean cost cannot replace the individual baseline. The nominal T196 conversion multiplies this ratio and its already-reported interval endpoints by 6000/5948. It is not a native recount.

The separate scramble file stores mean cost and corrected MSE/(p(1-p)/mean cost), with clipped RMSE, only. Its internal algebra can be checked. Individual costs/estimates and their covariance are unavailable. Jensen's inequality makes p(1-p)/mean cost no larger than mean[p(1-p)/cost]; a nonnegative numerator therefore gives a ratio at least the own-cost ratio. Negative corrected MSE requires retaining its sign rather than describing it as a conservative error bound. A 16-rule mean MSE neither certifies the selected fixed rule nor scores their averaged estimator at its total cost.

The one-model, one-run orig bootstrap resamples the same record and is degenerate. Subtracting truth variance does not propagate finite-reference uncertainty. Grouped bootstrap intervals retain their original unverified coverage; pooled-table generation is absent from this scoped driver. Current source pins do not recover historical weights, external libraries or source identity, nor the original historical sci_tao run.

The wording “no gain at sigma>=.005” is contradicted by the author's own rounded table. The supported scope is no consistent gain across these finite tested pairs. “Cannot be enumerated exactly” is not established by finite cell-growth counts. Neither deterministic-family impossibility nor a challenge solution follows.

## Saved-checker review
The frozen checker reads only JSON/text bytes with stdlib. Exact Fraction contractions use stored binary64 values; the fixed absolute/relative tolerance compares author rounded summaries, and no tolerance adapts to outcomes. Coverage is two methods, 30 models, five budgets, 300 unique keys; 60 grouped rows, 10 pooled rows and 24 scramble aggregates. It retains original CIs and times without pretending to validate them; orig degeneracy and endpoint unit conversion are separately checked. Signed corrected MSE is preserved before clipped RMSE consistency. No bootstrap, RNG or author import is executed.

Required pre-execution hardening: an explicit completed independent review/publication and mandatory file-pin membership, rather than an arbitrary possibly empty pin list. This is a provenance correction, not an arithmetic change. Clear only after the final hardening is inspected and bound in source_gate.json. No score execution has occurred in this review.
