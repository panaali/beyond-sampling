# T226 — independent T223 evidence inventory

## Question
What does the published blind-stratified tie-slicing result actually establish, and what remains unverified?

## Method
Source-only review of author commit `2809650d`, claimed in TASKS-only `b15709bc`. `manifest.json` records original Git blobs, SHA256, sizes and paths for the owned `frozen/` snapshot. `snapshot.py` copies Git objects without importing them. The full historical leaderboard raw blob is pinned without a second copy; the scoped author raw file is preserved. Source inspection covers the local estimator, face walker, score/scramble drivers, leaderboard and model-loading dependencies. External installed AlgZoo/library/checkpoint bytes and historical runtime identity remain unbound.

## Status
Source/proof/provenance phase complete; independent review requested. No numerical score contraction, estimator replay or shared adoption. Original author ancestry is available locally but is distinct from this published owned snapshot. Any saved arithmetic phase requires a separate frozen scope and source gate.

## Results
See `SOURCE_FINDINGS.md`. The exact event probability is 11/1024, while its conditional error is estimated by a fixed Sobol rule. The nonblind clipping formulas agree with the two rank charts. Floating traversal, cost and evidence qualifications remain material. The claimed 16-scramble improvement is an aggregate reported observation with no saved per-scramble values or confidence interval. It is not a deterministic accuracy guarantee or a mechanistic result.

Reproduce the metadata snapshot from a clean owned folder using `python review/tie_blind_evidence/snapshot.py` with the author Git object present. It refuses an existing manifest. This is a metadata operation, not an estimator run. Source findings refer to files within `frozen/` and require no model execution.

## Compute accounting
Only Git reads, hashing, byte copies and textual inspection. Zero author imports, model forwards, checkpoint deserializations, fits, RNG draws, MC, Gaussian quadrature or geometry. No estimator FLOP estimate is inferred from review work. No numerical experiment ran and no numerical outcome was selected.

## Claims
- [inferred, source/theory] Independent signs give P(B)=(T+1)/2^T; the stated nonblind fibre restrictions follow from the second order statistic.
- [inferred, source] The implementation estimates the blind conditional error and uses floating face tables/walks; it is not exact blind integration.
- [measured, source inventory only] The pinned `seeds.py` writes only mean cost, corrected mean-square ratio and clipped RMSE for each model/budget/method, not individual scramble values or costs.
- [inferred] These finite results do not establish a full challenge win, actual full-cost advantage, or impossibility of other tie-slicing variants.

## Alternatives not ruled out
A better partition/allocation, an exact or certified blind subproblem, stable/certified fibre traversal, reduced traversal work, more informative recorded scramble results, and stronger truth/error bounds remain open. An apparent gain can reflect rule variability or incomplete cost accounting; the cheapest discriminators are saved arithmetic verification and a source-specific cost inventory, respectively. No new numerical campaign is proposed here. A real split benefit and a noisy scramble estimate are compatible.

## Failure modes
Historical source/checkpoint identity cannot be inferred from a current Git snapshot. The original walker can stop at a cap without recording residual mass and uses floating epsilon/tie tolerances. Single-model one-run bootstrap intervals are degenerate. Scramble averages do not certify the chosen fixed rule. Costs inherit a correction from a related implementation. Each of these limits is preserved rather than treated as an observed trained failure.
