# T197 E43 cell-statistics provenance recovery

## Question
Which published cell counts include rescaling-test work, and which exact
corrections can be recovered without evaluating models again?

## Method
Claim321fdd66 pushed first. Freeze original evalH/reportH/est75 and all four
E43 summaries; separate value/operation counters from mutable STATS. Own
copy evalH_fixed.py and proposed evalH_stats.patch snapshot statistics before
the rescale check. Shared implementation remains DRI-owned. Next fixed
synthetic extracted-loop controls exercise unequal rescaled counts and confirm
unchanged values/ops/invariance checks; no estimator/model imports or calls.

## Status
Bounded source/controls phase complete. Published claim6928abad and protocol de00d243 followed independent source gatea2465c24; six frozen synthetic controls212252f5 pass, independently audited6c3bdd74. Shared evaluator adoption and historical-table reconstruction remain separate DRI decisions. No actual model/evaluator/truth/random call was made.

## Results
Six fixed extracted-loop cases pass. Each original row retains cumulative original-plus-rescaled counts when selected; each repaired row retains only original counts. All estimates, truths, original operation counts, exact operation counts, leaves, rescale invariance values, call ordering and final STATS remain identical. Independent review reconstructs every saved row/event without invoking author code.

Reproduction (one immutable attempt, already executed):
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python review/evalH_stats_review/controls.py`, with external `ulimit -v 2500000`. The runner deliberately rejects an existing controls.json; inspect saved evidence instead of overwriting it.

The source appends rows AFTER its every-eighth rescale evaluation, using
cumulative E75.STATS. H4 seed2 has historical clean_cells297, proving that
original/rescaled integer cell counts cannot always be equal. Halving every
clean count is therefore invalid. Noisy inflation is the sum of selected
rescale counts divided by256, not necessarily one eighth of the mean.
Historical E43 files store aggregates only. T113 fresh rows contain scalar
rec112 statistics (different implementation/noise panel), so those cannot
silently replace missing native statistics.

## Compute accounting
Read-only source/JSON/header inventory and source transformation only. No model,
checkpoint load, geometry, mass, truth, Monte Carlo or native recount calls.
Fixed controls ran once under review lock,19/20CPU cap and external2GB/one thread:0.016294550CPU. Independent source and saved reviews used0.015974761 and0.003998CPU separately. The53 synthetic row pairs involved128 fake estimator calls,106 fake truth calls and22 fake rescale calls; none is a model forward or probability evaluation.

## Claims
[inferred, independently source/control reviewed] The reporting bug leaves saved estimate,
truth and the original Ops object unchanged, but contaminates cells/cut/resolved.
[measured source inventory] Exact original percopy statistics are absent from
the historical aggregate summaries. Broader artifact search remains open.

## Alternatives not ruled out
Other saved matching-model native statistics may recover clean cases; carefully
scoped reproduction may recover missing selected rescale counts later. Pure
source count invariance is insufficient because floating branch decisions can
differ. Growth extrapolation remains empirical, not a lower-bound theorem.

## Failure modes
Blindly dividing by2 or9/8; mixing scalar/native topology, noise seeds70 vs80,
model versions or recipe variants; inferring cell-count equality from close
output values; treating sampled-region lower bounds as full region counts.
Original summaries and all failed controls will be retained.

## Saved native H4 clean evidence located

Prior independently reviewed T115 continuation_pilot contains native RECIPE112 cells174,127,148,150,175 for clean seeds0..4, with matching original operation counts69392,63026,57598,64665,72537. Historical E43 cells were348,254,297,300,350. The seed2 mismatch is148+149, not an exact double. The frozen source is identical after removing only two provenance comment lines. This binding still needs the current independent saved-contraction gate; no corrected full-table/scaling result is claimed.

No estimator, model input, truth, geometry or Monte Carlo call was made in this phase. Proposed remaining15clean/original5120-noise statistics reconstruction has NOT been implemented or authorized for execution; it needs a new published resource scope after review/publication resumes.
