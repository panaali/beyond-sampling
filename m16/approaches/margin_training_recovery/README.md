# T115: margin training-axis recovery

## Question
What was actually completed for frozen RECIPE85B/RECIPE112 on the training axis, what can be independently reproduced from saved evidence, and which next rung/panel is still missing?

## Method
Claim aab3f0f2 precedes implementation. First freeze existing source and margin_k28/k30 plus corresponding T101 truths; preserve all results and gaps. Independently reconstruct saved RMS/bias/max and p20 aggregate error ratios from paired truths using both ordinary fsum and exact decimal/rational contractions. Reproduce original pointwise bootstrap only where all inputs are saved. Do not infer missing p100 second moments from mean truth. Cost values are inherited source counters, not recertified by this inventory.

Source est75_t112.py differs from T113 pinned est75.py only by two provenance comments: stripped SHA426020c67a52678e5cccb20c552837fe2d233c3fe5e13557916034a3c46877d3. Coordinate T113's active k30 source recount, avoiding duplicates. It uses an explicit numeric convention with sampler114 for H4; T114/T121 protocol112 excludes3 model-free rank comparisons and includes1 correctness comparison. Numerator conventions must also align before converting full budget claims.

## Status
Inventory frozen. Existing k28 contains all5 seeds×1024 p20 copies for both recipes; k30 contains onlyseed0 with1024 p20 and512 p100 for each recipe plus freshT75. No new model run yet.

## Results
Raw contraction and independent review pending. Fresh p100 saves error arrays and aggregate mean truth/mean counter but no individual truth/counter arrays. Its full ratio/CI is not independently recoverable from those summaries alone. Training amount k29 is absent from the original margin driver despite the one-step ladder rule; preserve that gap before extending below already savedk28.

## Compute accounting
First phase is read-only source/data inventory and finite raw arithmetic only. No model forward, checkpoint download, exact mass, new noise or sampling. One thread2GB, <=20CPU review cycles; future model runs require a frozen bounded scope after inventory.

## Claims
[measured from file inventory] Two saved margin files exist with partialk30. [open] Source counter, fresh truth reconstruction, full rung completion and matched-compute conclusions require separate evidence.

## Alternatives not ruled out
1. Partial files could be mistaken for complete rungs: enumerate seed/panel/recipe counts.
2. Ratios may reflect inherited mixed cost conventions: preserve original72 and coordinate independent counters before replacement.
3. Agreement with author exact.py may hide degenerate ties: retain T114/T113 independent reference caveats; generic saved trained results are not automatically invalidated.

## Failure modes
Repeating completed panels; replacing E43/T115 source snapshots silently; treating mean truth as mean p(1−p); inventing missing per-copy counters; skippingk29; claiming a full certificate from empirically accurate recipes after T108's bounded soundness finding; or reporting inherited operation counts as hardware/full implementation costs.
