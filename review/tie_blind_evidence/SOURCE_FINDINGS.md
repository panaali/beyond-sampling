# Source and proof findings (no numerical replay)

All author paths below are relative to `frozen/`; their exact bytes are in `manifest.json`. This phase does not independently certify the saved numerical ratios.

## 1. Exact event versus approximate error
For independent symmetric continuous inputs, every sign orthant has probability 2^-T. The event B that the second-largest input is nonpositive consists of the all-negative orthant plus T one-positive orthants. Thus P(B)=(T+1)/2^T=11/1024 at T=10. Conditional on B, these orthants are uniform; independent half-normal magnitudes give the ideal conditional input distribution. This establishes the weight, not the conditional probability that this network is wrong.

`tie_blind/core_nb.py::blind_points` implements the ideal map with a finite scrambled Sobol rule and endpoint clipping. `blind_wrong` evaluates the network on those points. Therefore the blind contribution is numerical quadrature of a model-dependent indicator. The author estimator docstring's claim that the blind set “cannot be enumerated exactly” is not a theorem supplied by finite prefix counts; finite growth only supports a cost concern for the tested construction.

For the rank-2/3 chart, the second-largest coordinate is u+|t|/sqrt(2), so the nonblind region is all t if u>=0 and |t|>sqrt(2)|u| otherwise. For the rank-1/2 chart it is u-|t|/sqrt(2), so the nonblind region is empty if u<=0 and |t|<sqrt(2)u otherwise. These are exactly the restrictions expressed by `walk_mass_nb` (endpoints have zero continuous Gaussian mass). This conclusion is conditional on the parent co-area chart decomposition and does not independently prove its normalization.

The chart sign u<=0 is not itself the input event B along an entire fibre. T218 face-sign summaries must not be silently substituted for the B probability/error decomposition.

## 2. Floating numerical and tie scope
The parent `sci_tao/core.py` builds its face CDFs using a finite grid [-8,8] and cumulative sums; this is not exact face sampling or certified normalization. `walk_mass_nb` uses eps=1e-10, finite roots and max_cells=2000. It returns mass/cost/cell counts even if active fibres remain at that cap; it exposes no residual-mass or completion flag. No cap occurrence is inferred for saved trained runs.

First-index output ownership is attempted through earlier-competitor handling. The `bad0` predicate also treats |c|<=1e-12*(1+max|L|) as an identity when slope e==0; this is a numerical tolerance rather than exact coefficient equality. The direct blind classifier uses NumPy argmax, which owns an exact tie by first index, but its input-label argsort does not specify a first-index tie policy. Input ties have zero mass under the ideal continuous law; a finite node map is a separate question. No new tie fixture or trained failure was run.

## 3. Cost is conditional, not closed
Each cell recomputes recurrent h and dh from zero for every time step, while the raw walker increments 2*(T-restart)/T plus a first-prefix charge. The local estimator applies the inherited factor2.23. This is a source-specific inherited correction, not a proof that every restriction, readout, setup, allocation and traversal operation is priced. Skipping wholly blind fibres changes the mix of work. Reusing a correction measured for the parent needs an applicability argument before an actual-cost claim.

`score.py` applies the T196 restatement ratio*6000/5948. That is a declared nominal unit conversion, not native-cost closure. `seeds.py` reports the historical FPE ratio without this additional factor. The two scales should be named separately. Model-free source imports also allocate a 400000-input reference array in arc_eval; no such import was executed by this review. The recorded author run command uses 2500000 KiB virtual memory, which is not a strict2GiB limit; no historical RSS or violation is inferred.

## 4. Saved evidence and inference limits
`seeds.py` uses six listed models, two budgets, two methods and sixteen scramble seeds. It subtracts saved truth_se^2 from each squared error, then divides mean corrected squared error by p*(1-p)/mean(charged cost). `seeds.json` stores only that ratio, mean cost and clipped RMSE. Individual estimates, cost associations, per-scramble states and confidence intervals are absent. The own-cost mean baseline mean(p*(1-p)/C_s) differs from p*(1-p)/mean(C_s); equality requires constant costs. By convexity the latter denominator is no larger, so this specific difference is conservative for a nonnegative numerator, but its magnitude is unrecoverable from the saved aggregates.

A mean across sixteen scrambled rules describes those rule realizations. It is neither the error of a selected fixed rule nor a guarantee for the estimator that averages all sixteen outputs (which would require its own total cost and covariance). The script gives no uncertainty interval for its estimated mean squared error. Its module comment that rule seeds are “not model inputs” does not change the estimator's quadrature class: those seeds determine the input nodes.

`leaderboard.table` resamples models and runs with a fixed bootstrap seed. An orig group containing one deterministic run resamples the same sole record, so its bootstrap interval is degenerate. Subtracting truth_se^2 corrects an expectation under suitable independence; it does not propagate the finite-reference uncertainty into that degenerate interval. The truth source includes32M reference sizes, not the later required100M full-rung protocol. No new reference or CI has been computed here.

The pooled table-generation source is not present in the scoped T223 files: score.py invokes the ordinary grouped table and writes table.json, not table_pooled.json. The raw rows may permit an independently defined pooled contraction later, but original pooled bootstrap provenance is incomplete. `run.log` and source bytes do not bind historical checkpoint/library/source identities. `seed_sci_tao` copies historical parent rows; original-time identity cannot be inferred from the current parent source.

## 5. Wording corrections supported by the existing table
The author's own 16-scramble table shows a gain at s0.005_0/64k (0.92 versus2.15) and a smaller gain at s0.02_0/64k (3.10 versus3.38). Therefore “no gain on sigma>=.005 copies” is too broad even as a description of that table. “No consistent gain across the tested noisy model/budget pairs” is the supported wording. These are transcriptions of author values, pending independent saved arithmetic review.

“Exact blind handling” should mean exact event weight and analytic nonblind interval restriction, with numerical blind error and floating fibre integration. “Best honest deterministic number” should be reported as the finite16-scramble quadrature MSE observation, with its missing CI and conditional cost/reference qualifications. Neither a broad tie-slicing impossibility nor a challenge solution follows.
