# T204 direct SRB cost prerequisite

## Question

Does the implemented stratified Rao–Blackwell estimator justify the historical
3.3/1.6 cost correction under the current leaderboard convention? This phase
pins the call path and a bounded control plan before measuring anything.

## Method

Source-only snapshot from sharedff71bdaa, following explicit yield by910 and
TASKS-only claimff71bdaa. manifest.json records thirteen original paths, Git
blobs, SHA256 hashes, byte sizes and Python definitions/imports. No author module
was imported; model, estimator, Monte Carlo, Gaussian draw and native-kernel
calls are all zero. SOURCE.md gives the observed call path. ACCOUNTING.md fixes
which symbolic work is known and which still needs instrumented validation.
CONTROL_PLAN.md is a proposal, not an execution gate.

## Status

Source/provenance and analytical prerequisite prepared; independent review
pending. No claimed replacement multiplier or trained cost measurement.
Existing T204 saved all-budget audit b4093511/e353e6e6 remains unchanged.

## Results

[inferred from pinned source] The entry builds seven-stratum, no-control-variate
`cv.make(strat=True, cv=None, alloc=plain)`. Prefix allocation uses1.7FPE, at least
four kept rows per stratum. The reported count is ceil(1.7M). The harness then
multiplies that whole count by3.3/1.6, including the original draw allowance and
rounding. That differs from replacing only the1.6 Q allowance by3.3.

[inferred from pinned source] `Q` recomputes two dense logit products for every
interval with at least one active batch row; it does not perform the comment's
rank-one updates. Its loop has H+3 intervals for H recurrent units, not uniformly
18. Whole-batch interval execution prevents transferring a single-row trace to
an arbitrary batch. `draw_all` returns a candidate count that the estimator
throws away, so its fixed0.1 allowance does not directly enforce the current
extra-candidate0.04FPE rule.

[inferred from pinned source] The generic exact-conditional-accuracy docstring
also needs a tie qualification: all zero readout rows force the implementation
to return zero, although first-index argmax chooses label0 and has positive
accuracy. The cost audit must preserve this behavior in parity controls rather
than silently repair it. Shared repair is a separate backlog item.

## Compute accounting

This phase only reads Git objects, hashes bytes and parses ASTs. No numerical
kernel or estimator cost has been measured. Universal and original-model advice
are separately identified. T196 scalar units and leaderboard FPE rules are
recorded without treating approximate counts as complete FLOPs or hardware cost.

## Claims

Only the source facts above are asserted. [speculative] At large batches, union
of active interval positions can make the old3.3 estimate inaccurate. The fixed
control plan must test interval gating and count preservation before any trained
case. No reversal of an estimator ranking follows from this inventory.

## Alternatives not ruled out

The hand multiplier could approximately fit a particular trained batch even
though the assumed incremental implementation is absent. The draw allowance
could be adequate at some budgets and inadequate at others. Native library work
and source-level abstract arithmetic need not coincide. Numerical tie or tiny
slope errors could affect estimates independently of cost. None is resolved by
source inspection alone.

## Failure modes

Historical runtime imports and input batches are not authenticated by a current
source snapshot. Sorting, special functions and library internals remain open.
Zero-output ties and the1e-12 slope cutoff limit universal correctness claims.
The current source/control plan must not be used to rerun the broad leaderboard
or apply a fitted multiplier without a new published bounded gate.
