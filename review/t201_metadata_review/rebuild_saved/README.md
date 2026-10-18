# T201 independent saved synthetic-control review

## Question
Do the published seven synthetic groups preserve their fixed coverage, provenance and analytic invariants?

## Method
check.py imports no author code and makes no numerical solve or author forward call. It checks exact fixed state keys, singular-value trace/product invariants, saved residual thresholds, domain guards, metrics, independently sorted global-rank cross-tabs and six exact scalar recurrences. The coefficient/prediction arrays were not serialized: the residual maxima are source-gated author summaries, not independently recomputed fits.

## Status
Complete. Assignment34b1bc00, source0073f39a hardenedc8e93b53, independent source gateeca92ba8/shared977c6b5c, execution pinsad05f940/shared5f3e64c4 preceded the sole saved-only audit. Initial author launcher exit127 occurred before Python and is preserved; corrected run2 alone produced scientific records.

## Results
[measured] All seven groups,216 legal states,864 key records,219 recorded solves,6 synthetic sequences and complete provenance/progress pass. Exact recurrence outputs and 9 cross-tab rows independently match. Author CPU.219838866; reviewer parent CPU.005030145, Git child overhead excluded. Saved numerical residual maxima are below frozen1e-10; source/meta checks support those records without re-solving.

## Compute accounting
One <=20CPU/2GB review-lock invocation: saved JSON/source hashing, Git metadata and exact scalar arithmetic only. No author import, least-squares solve, random input or trained/model execution. Six short scalar synthetic recurrences are reviewer oracle arithmetic, explicitly not an author forward replay.

## Claims
[measured] Fixed saved coverage and analytic oracles pass. [inferred] This supports the bounded helper-control handoff. No checkpoint wiring, historical OOS score, new trained performance or challenge criterion is established.

## Alternatives not ruled out
Actual model wiring and held-out data behavior require separately frozen work. Serializing coefficients/predictions would allow a stronger saved residual audit, but is not retroactively manufactured here.

## Failure modes
Treating source-gated residual summaries as independent refits; interpreting synthetic controls as a trained accuracy claim; hiding the pre-Python launch failure. Source-only fixture-coverage hardening occurred before the sole reviewer run and is preserved in history.
