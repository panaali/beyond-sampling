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
Source/provenance inventory and patch preparation. No controls executed.

## Results
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
Fixed controls require independent source gate and capped review process.

## Claims
[inferred, independent review pending] The reporting bug leaves saved estimate,
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
