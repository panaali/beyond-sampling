# T226 — independent T223 evidence inventory

## Question
What does the published blind-stratified tie-slicing result actually establish, and what remains unverified?

## Method
Source-only review of author commit `2809650d`, claimed in TASKS-only `b15709bc`. `manifest.json` records original Git blobs, SHA256, sizes and paths for the owned `frozen/` snapshot. `snapshot.py` copies Git objects without importing them. The full historical leaderboard raw blob is pinned without a second copy; the scoped author raw file is preserved. Source inspection covers the local estimator, face walker, score/scramble drivers, leaderboard and model-loading dependencies. External installed AlgZoo/library/checkpoint bytes and historical runtime identity remain unbound.

## Status
Bounded source and independent saved-arithmetic review complete. Independent source/proof/manifest review6907d022 (shared eb52ecaf) cleared the frozen checker0434255f after explicit gate hardening; immutable execution gate a05c510c was published at fa1567fe before the sole run. No estimator replay or shared adoption. Original author ancestry is distinct from this published owned snapshot. Further source-cost, uncertainty or estimator experiments require a new separately prioritized scope.

## Results
The saved-only checker passed all300 unique raw records,60 grouped scores,10 pooled scores and24 aggregate consistency checks. Source-table ratios (historical FPE convention) for tie_blind pooled across30models are6.53694,4.66829,3.98402,3.35872,3.91480 at1k,4k,16k,64k,256k; all exceed1. The chosen fixed rule on orig gives0.47512 at16k and2.61500 at64k, which is distinct from the reported mean over16scrambles. Original CIs remain inherited; no bootstrap was regenerated. `saved/results.json` retains exact rational contractions and every original table/aggregate record. See `SOURCE_FINDINGS.md`. The exact event probability is 11/1024, while its conditional error is estimated by a fixed Sobol rule. The nonblind clipping formulas agree with the two rank charts. Floating traversal, cost and evidence qualifications remain material. The claimed 16-scramble improvement is an aggregate reported observation with no saved per-scramble values or confidence interval. It is not a deterministic accuracy guarantee or a mechanistic result.

Reproduce the metadata snapshot from a clean owned folder using `python review/tie_blind_evidence/snapshot.py` with the author Git object present. It refuses an existing manifest. This is a metadata operation, not an estimator run. Source findings refer to files within `frozen/` and require no model execution.

## Compute accounting
Git reads/hashing/source inspection plus one saved-only rational contraction:0.151590305 internal CPU seconds,26712KiB peak RSS,95 durable records. Imports are excluded from the internal timer. External20CPU and2097152KiB virtual caps were enforced under the review lock. Zero author imports, model forwards, checkpoint deserializations, fits, RNG draws, MC, Gaussian quadrature or geometry. No estimator FLOP estimate is inferred. No numerical estimator experiment ran or outcome was selected.

## Claims
- [inferred, source/theory] Independent signs give P(B)=(T+1)/2^T; the stated nonblind fibre restrictions follow from the second order statistic.
- [inferred, source] The implementation estimates the blind conditional error and uses floating face tables/walks; it is not exact blind integration.
- [measured, independent saved arithmetic] All300 raw records and70 score rows satisfy the independently reconstructed own-cost formulas;24 scramble records satisfy aggregate-only RMSE/ratio/cost consistency. No missing individual scramble evidence was reconstructed.
- [measured, source inventory only] The pinned `seeds.py` writes only mean cost, corrected mean-square ratio and clipped RMSE for each model/budget/method, not individual scramble values or costs.
- [inferred] These finite results do not establish a full challenge win, actual full-cost advantage, or impossibility of other tie-slicing variants.

## Alternatives not ruled out
A better partition/allocation, an exact or certified blind subproblem, stable/certified fibre traversal, reduced traversal work, more informative recorded scramble results, and stronger truth/error bounds remain open. An apparent gain can reflect rule variability or incomplete cost accounting; the cheapest discriminators are saved arithmetic verification and a source-specific cost inventory, respectively. No new numerical campaign is proposed here. A real split benefit and a noisy scramble estimate are compatible.

## Failure modes
Historical source/checkpoint identity cannot be inferred from a current Git snapshot. The original walker can stop at a cap without recording residual mass and uses floating epsilon/tie tolerances. Single-model one-run bootstrap intervals are degenerate. Scramble averages do not certify the chosen fixed rule. Costs inherit a correction from a related implementation. Each of these limits is preserved rather than treated as an observed trained failure.

## Saved audit reproduction
The sole executed command was `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash -c 'ulimit -v 2097152; ulimit -t 20; flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python review/tie_blind_evidence/saved/check.py' > review/tie_blind_evidence/saved/run1.log 2>&1`. The checker refuses existing outputs; do not rerun in place. Source review, protocol and all pins remain in `saved/gate.json`.
