# T259 independent written margin-loss review

## Question
Do the frozen identities and their source specialization justify eliminating a redundant common-design quadratic-loss substitution?

## Method
Read producer `3bc9e2c3` under delegated claim `92a67233`. Independently derive the two matrix identities and inspect the five exact text snapshots at `a9b09a14`. `check_metadata.py` checks lengths, SHA256, Git blob identities and package bytes without importing the author. `source_review.json` records the written gate.

## Status
PASS for the written algebra, source specialization and provenance. Execution is not authorized. No fitted or empirical conclusion is independently reproduced.

## Results
[inferred] For a common feature penalty and invertible normal matrix, fitting is the same left-linear map applied to every target column. Right multiplication by any fixed readout therefore commutes with the fit and query prediction. A common fixed generalized inverse also suffices in a singular problem; independently selected minimizers do not. The identity survives fixed cell routing and a common pooled fallback. A training-derived feature bank must be frozen identically in both arms; refitting it is outside the identity.

[inferred] Expanding the unordered margin square sum collects each squared error K-1 times and subtracts twice every cross term. This equals K times the squared norm after class centering. Ordered sums double it. Dividing either by its number of pairs yields the factor 2/(K-1). Sample/class normalization must be tracked separately.

[inferred] Write B=D+b1^T with D1=0. Both residual and shared feature-penalty cross terms vanish because the centered columns sum to zero. The ordinary ridge centered minimizer is B0C. Multiplying the centered data term and the whole-B penalty by the same positive factor leaves that centered minimizer unchanged; the common-mode penalty is separate. Its possible nonuniqueness changes no margins at any query. Equivalently one may penalize only D and fix the gauge. Argmax and the full tie set are unchanged in exact arithmetic, so first-index ties are also unchanged. Holding the numerical penalty fixed while changing data normalization is a regularization change, not this matched comparison.

[inferred] The written separating cases correctly identify missing assumptions: output-specific penalties/features, unmatched intercepts, arbitrary null-space solutions and nonlinear clipping need not commute. A fixed readout need not be square or invertible. New weighted/top-rival/nonquadratic objectives or feature changes are not ruled out, nor guaranteed to be useful.

Source inspection confirms:

- `CellLinear.fit` forms target-independent cell Gram matrices and applies one design-scaled scalar ridge to every target column. The pooled matrix uses the stated separate design-only penalty. Its zero-design singularity is not repaired or ignored; successful solves or a specified common linear inverse remain a premise.
- `predict` selects seen cells and fallback from input cell IDs, without selecting by output targets.
- `HingeBank` uses training comparator preactivations and then ReLU. Its frozen query features are input-computable; training advice and feature construction costs remain separate.
- `phaseC_diag.py` uses the same cells, designs, bank, default penalties and readout for direct logit fitting versus post-ReLU hidden fitting. The latter query predictions are multiplied by the readout without another clipping. Exact-real commutation applies conditionally; target formation, accumulation, solves and final multiplication may differ in finite precision.
- `phaseC2.py` selects per-neuron/time families and clips replacement values with ReLU. Its common-design equality is not established by this theorem. Test-selected cumulative ordering also remains a separate evaluation-design issue covered by the prospective protocol.

Reproduce provenance only:

`python review/t259_margin_loss_review/check_metadata.py`

No numerical experiment command or empirical tolerance is supplied.

## Compute accounting
Only source-text reads, hashes and Git byte comparisons. Zero author imports, arrays, checkpoints, model/fit/forward/RNG calls, numerical controls or score contractions. No scientific CPU total, native operation count or deployment cost is claimed.

## Claims
[inferred] The specific shared-design, matched-ridge unweighted all-pair margin substitution is redundant in exact arithmetic. [inferred] The pinned direct-logit/post-ReLU-hidden diagnostic has that structure, subject to successful common solves. No measured performance or bitwise parity finding is added.

## Alternatives not ruled out
Objective mismatch may be addressed by genuinely different losses; feature insufficiency may be addressed by new features or a different architecture. The written identity distinguishes a redundant substitution but does not decide between those explanations, prove an information floor, or rule out 99% agreement.

## Failure modes
Unmatched penalties or normalization, target-dependent refitting, output-specific routing/features, different singular solution conventions, nonlinear postprocessing and finite arithmetic prevent broader transfer. Source bytes alone do not validate historical runtime, trained results, future held-out selection or retained coefficients. T258 reproduction and T260 description length remain separate scopes. No new external ARC/AlgZoo source was consulted.
