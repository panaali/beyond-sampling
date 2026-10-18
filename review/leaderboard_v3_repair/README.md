# T204: leaderboard source and saved evidence repair

## Question
Can the existing evidence be reported without selecting the best budget, confusing untested determinism with failure, or mixing E44 and T114 cost denominators?

## Method
`manifest.json` pins fourteen source/data snapshots including the raw archive before the arc_cumulants classification addition. `audit.py` imports no author code. It checks five determinism metadata cases and an exact cost-covariance counterexample, inventories saved raw flags/history, exports every archived H16T10 row unchanged with separately labeled current DRI classification, and contracts the existing E44/T114 records with exact rational arithmetic. It does not regenerate bootstrap intervals or time methods. `PATCH_PROPOSAL.md` is an adoption proposal; the shared harness remains DRI-owned.

## Status
Fixed saved audit complete after independently published source gate ee41dae8. Independent saved verification e353e6e6 (shared44b10f30) passes all records/contractions; shared adoption remains DRI-only. No estimator, forward, truth, Monte Carlo, geometry, or native-cost calls occurred.

## Results

The one gated run `cb980799` completed in0.699786896CPU seconds. All498 unique archived budget/group/method rows retain every original field;50 inherited intervals are degenerate, no negative ratios occur. Both raw snapshots have7,896 conflicting overwritten records, all explicitly inventoried under the original latest-key policy. All2,880 selected arc_cumulants records are identical to the pre-class snapshot and have legacy False; current DRI class is mechanistic, prior source class unreviewed. This establishes archival precedence, not runtime provenance or a fresh determinism pass.

Exact saved-float contractions reproduce E44 .5883302365250083 (1024copies, mean proxy/72, archivedCI[.5129496132,.6614431228]) and T114 .7651127844230149 (256copies, own scalar/112, archivedCI[.5802543086,.9540812837]). On the T114 panel alone, uniform mean-cost scoring gives .7670187955711592, showing that allocation convention alone is small on this particular panel; cost-count and noise-panel differences remain distinct. Five metadata controls and the rational covariance control pass. The latter has identical errors/truths/mean cost but ratios1/10 versus3/26 under swapped cost vectors, versus1/7 uniform: E44 missing per-copy costs cannot be uniquely reconstructed.
Source findings: current `table()` and `report()` already iterate all budgets, while `results/leaderboard_v3_summary.txt` displays one chosen budget per method/group. The summary generator is not present in the reviewed sources. `run()` initializes det=False for unreviewed/sampling methods and saves that same field when no test ran; `table()` subsequently demotes a current mechanistic/quadrature method if any historical record has False. Therefore False alone cannot distinguish absence of a test from a failed test.

E44 k24 seed4 has 1,024 absolute-noise copies and mean-counter/72 uniform-budget scoring. T114 has 256 different copies (noise120/121), independent scalar counts/112 and own-model budgets. E44 saves truths/errors and only mean estimator cost; its per-copy cost/truth covariance is unavailable from these fields. Neither result should silently replace the other.

Reproduction, only after the independent source gate is pinned in gate.json:
`ulimit -v 2097152; ulimit -t 20; OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python review/leaderboard_v3_repair/audit.py`
The script refuses to overwrite its first results artifact and preserves partial failures. A failure requires a documented repair/gate rather than deleting the artifact.

## Compute accounting
Fourteen byte snapshots plus one saved arithmetic audit,0.699786896CPU under19/20CPU and2GiB cap; no stochastic draws (including bootstrap), model evaluations or author imports. Archived CPU and precomputation figures are inherited evidence, not measured by this review. srb3.3/1.6 remains a hand-count multiplier, not instrumented here.

## Claims
- [measured] All498 exported rows preserve original fields;2,880 arc records exactly match the pre-class archive. Independently verified e353e6e6.
- [inferred] Legacy False is ambiguous by the frozen run source. Explicit tested=False, tested-pass and tested-fail metadata would remove the ambiguity for future runs.
- [inferred] Existing all-budget rows can be preserved without calling report(), whose timing helper would otherwise evaluate sampling models.
- [inferred] Missing per-copy costs prevent a unique conversion of E44's uniform mean-cost ratio to T22's own-cost ratio; the frozen rational control tests this.
- [speculative] Historical identical records predating a class change explain the arc_cumulants label. Snapshot identity supports archival presence, not proof of which runtime source produced the record.

## Alternatives not ruled out
A genuine failed determinism check could also have emitted False; no unknown is promoted to pass. A separate summary producer may exist outside reviewed history. Cost reweighting and sampling variation can both change E44/T114 comparisons; different panels preclude isolating one cause. An instrumented srb count may support or reject the historical multiplier, but is a separate scope.

## Failure modes
Archived intervals are copied and can be degenerate; they do not exhaust truth uncertainty. Pointwise/bootstrap success is not directive7 simultaneous rung validation. Class membership is DRI judgement, not inferred from identical outputs. Exact contractions of stored binary64 data do not make original model arithmetic exact. No full-cost or general method rejection follows.

Source-gate harness correction: explicit fixed counts, source/manifest bindings, equal-length positive-cost rational inputs, immutable report paths, asserted field preservation and complete duplicate-key replacement inventory were added before any execution. Latest-record semantics remain the pinned author policy; duplicate records are not silently discarded in the audit.

Independent saved audit e353e6e6 checks all498 original fields/table rows, both7,896 replacement inventories, five metadata/one exact covariance controls, E44/T114 contractions and inherited CIs in0.673951434CPU without author calls. The later source-only srb addendum is distinct from this executed/checker scope.
