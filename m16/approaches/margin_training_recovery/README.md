# T115: margin training-axis recovery

## Question
What was actually completed for frozen RECIPE85B/RECIPE112 on the training axis, what can be independently reproduced from saved evidence, and which next rung/panel is still missing?

## Method
Claim aab3f0f2 precedes implementation. First freeze existing source and margin_k28/k30 plus corresponding T101 truths; preserve all results and gaps. Independently reconstruct saved RMS/bias/max and p20 aggregate error ratios from paired truths using both ordinary fsum and exact decimal/rational contractions. Reproduce original pointwise bootstrap only where all inputs are saved. Do not infer missing p100 second moments from mean truth. Cost values are inherited source counters, not recertified by this inventory.

Source est75_t112.py differs from T113 pinned est75.py only by two provenance comments: stripped SHA426020c67a52678e5cccb20c552837fe2d233c3fe5e13557916034a3c46877d3. Coordinate T113's active k30 source recount, avoiding duplicates. It uses an explicit numeric convention with sampler114 for H4; T114/T121 protocol112 excludes3 model-free rank comparisons and includes1 correctness comparison. Numerator conventions must also align before converting full budget claims.

## Status
Raw recovery complete and independently reviewed (23c5c972). Existing k28 contains all5 seeds×1024 p20 copies for both recipes; k30 contains onlyseed0 with1024 p20 and512 p100 for each recipe plus freshT75. No new model run yet.

## Results
Exact-rational contractions reproduce all 12 paired p20 rows and 12 clean rows; all 1000-resample pointwise bootstrap intervals reproduce. Three p100 rows reproduce only saved RMS, bias and maximum error. The largest saved p20 ratio is 0.103000347197324 and largest upper endpoint is 0.114535771533307, under the inherited 72-operation convention. Independent 70-digit Decimal and separately sequential bootstrap review 23c5c972 agrees. This bootstrap holds the saved mean cost fixed; individual cost pairing is not recoverable. Fresh p100 saves error arrays and aggregate mean truth/mean counter but no individual truth/counter arrays. Its full ratio/CI is not independently recoverable from those summaries alone. Training amount k29 is absent from the original margin driver despite the one-step ladder rule; preserve that gap before extending below already savedk28.

## Compute accounting
First phase is read-only source/data inventory and finite raw arithmetic only. audit.py used 0.331433423 CPU seconds; independent review used 0.398 CPU seconds. No model forward, checkpoint download, exact mass, new noise or sampling. One thread2GB, <=20CPU review cycles; future model runs require a frozen bounded scope after inventory.

## Claims
[measured from file inventory] Two saved margin files exist with partialk30. [open] Source counter, fresh truth reconstruction, full rung completion and matched-compute conclusions require separate evidence.

## Alternatives not ruled out
1. Partial files could be mistaken for complete rungs: enumerate seed/panel/recipe counts.
2. Ratios may reflect inherited mixed cost conventions: preserve original72 and coordinate independent counters before replacement.
3. Agreement with author exact.py may hide degenerate ties: retain T114/T113 independent reference caveats; generic saved trained results are not automatically invalidated.

## Failure modes
Repeating completed panels; replacing E43/T115 source snapshots silently; treating mean truth as mean p(1−p); inventing missing per-copy counters; skippingk29; claiming a full certificate from empirically accurate recipes after T108's bounded soundness finding; or reporting inherited operation counts as hardware/full implementation costs.

## Frozen missing-k30 continuation
Published TASKS-only scope 4b6e3a4d. `continuation.py pilot` checks all five clean checkpoints, the first copy of each noise seed 20–23 for completed seed0, and the first four copies for each missing training seed1–4. Every pilot truth uses the independently reviewed T114 ordered-chart reference; original zero/identical-readout reference limitations do not disappear through source agreement. Seed0 recipe errors and clean native counters must reproduce saved rows. Checkpoint, dependency and model hashes are retained. The original absolute sigma .05 and per-matrix draw order are unchanged.

Only after these provenance checks pass, `continuation.py <seed> <noise_seed>` fills missing k30 seed1–4 p20 shards (256 copies each), reusing pilot evaluations. No completed seed0 panel is repeated. Noisy estimates and individual inherited native counters are saved; truths outside the pilot are the original T101 values. No input Monte Carlo or training occurs. The full frozen continuation has four missing seeds ×1024 copies×two recipes, with 64 noisy pilot pairs reused, plus 5 clean and 4 noisy seed0 control pairs. This is a recovered partial rung, not a full training-ladder or common-cost pass; k29 and new p100 remain uncomputed by this continuation. Numerator cost closure is a separate T113 successor owned by recover_910.

Reproduce each phase with one thread, 2GB and primary compute lock, <=110 CPU seconds per process. Pilot and shard files are never overwritten; interruption requires a separately documented resume preserving prior cost/evidence. `audit.py` reproduces the saved-data recovery without model evaluations.

## Recovered k29 dependency (read-only, not yet evaluated by margin recipes)
The original margin/T101 driver omitted k29, but a separate earlier T90 lane did not. Coordination with recover_56 located `training_axis_ladder/data/H4_T3_n2p29_seed0..4.pth`, manifest and all20 block files (5120 truths), independently reviewed e49308c. My first filename-based checkpoint search used the original `2nd_argmax_...` naming and missed these renamed copies; any initial statement that k29 checkpoints were absent locally is superseded. No download or repeated truth panel is needed. A fresh frozen scope and complete current k30 handoff precede any k29 margin evaluation.

## Completed missing k30 p20 panels
All four missing training seeds now have1024 original p20 copies each, with both frozen margin recipes. No copy was dropped or selected by outcome. The73-row pilot source/noise/weight audit is independently reviewed in6e892616. Independent final review a558c8a9 reconstructs all4096 noisy weight hashes, verifies all16 shards and64 pilot reuses exactly once, all8 panels/8 clean summaries and three bootstrap conventions. It preserves every individual loss; no estimator/truth rerun was needed. These measurements use the inherited native72 convention and do not certify whole-implementation cost.

| seed | recipe | original mean-cost ratio | fixed-mean-cost 95% interval | per-copy-budget ratio | individual losses/1024 |
|---|---|---:|---|---:|---:|
| 1 | r85b | 0.0432573025 | [0.0372665343, 0.0503815455] | 0.0429481722 | 1 |
| 1 | r112 | 2.61965564e-06 | [1.99021929e-08, 7.03569397e-06] | 2.60898754e-06 | 0 |
| 2 | r85b | 8.73134601e-05 | [7.12849831e-05, 0.000106136688] | 8.72008106e-05 | 0 |
| 2 | r112 | 1.27170369e-10 | [1.44589121e-12, 3.11234585e-10] | 1.27111849e-10 | 0 |
| 3 | r85b | 0.00871230811 | [0.00766481761, 0.00997543815] | 0.00864697909 | 0 |
| 3 | r112 | 0.002049995 | [0.00157783497, 0.00262765727] | 0.00204463521 | 0 |
| 4 | r85b | 0.160136431 | [0.133295475, 0.191064901] | 0.159571525 | 24 |
| 4 | r112 | 0.00363370662 | [0.00134251813, 0.00681376815] | 0.00362511166 | 1 |

All eight new clean ratios are below .039951; all aggregate pointwise upper endpoints are below .191065 under this inherited convention. This does not mean every copy wins: r85b seed4 has24 individual losses and worst ratio8.0354842; r112 seed4 has one loss, ratio1.0078639; r85b seed1 has one loss. Per-copy cost resampling gives maximum uniform-budget upper endpoint .1908789938, distinct from the original fixed-cost bootstrap. Intervals are pointwise and conditional on the frozen data/protocol, not a simultaneous or population certificate. Future corrected-cost results must be separately recomputed; T172 covers RECIPE112 and Ready T176 covers RECIPE85B.

Execution:16 serialized shards,199.434386952 CPU seconds; pilot4.702280374 CPU; contraction.243578846 CPU. Timers omit module imports before their start and do not claim whole-campaign cold cost. There are8210 new recipe evaluations, including9 completed-seed/clean control models and4096 newly filled noisy models;64 noisy pilot pairs were reused. There are73 independently evaluated reference models; remaining truths reuse T101. Model-input sampling and training are zero. Reproduce raw contractions with `python summarize_continuation.py` after all frozen shard files exist. No rerun is required for independent data arithmetic.
