# T115 k29 margin continuation

## Question
Do the two frozen margin recipes remain competitive on the missing k29 training level when costs are measured directly under the reviewed named-source convention?

## Method
Claim5be203e1. Reuse five T90 checkpoints and all5120 existing absolute.05 noisy-copy truths (independent e49308c). No training, MC or new truth. Pin all bytes, regenerate every noisy model hash, and use unchanged T176/T172 tracers for r85b/r112. Two fixed controls are seed0/noise20/copy0 and seed4/noise23/copy255; require baseline/traced output hex, legacy, STATS and grouped errors parity. Then first64 seed0/noise20 copies form the cost/parity gate before expansion. All failures preserved.

## Status
Full frozen continuation running after independent first64 reviewf4b76ba3 and source gate. All5120 provenance hashes pass. Two controls and five clean cases complete; remaining79 native64-copy shards run sequentially with fair lock release. No full-rung result yet.

## Results
First64 fixed seed0/noise20: all128 ledgers/parity pass; r85b/r112 numeric totals8,424,200/10,559,696 and per-copy matched111-budget ratios.000234968429/4.95786826e-7. No extrapolation to remaining seeds. Independent f4b76ba3. Commands: `python provenance.py`; after independent gate `python control.py 0` and `python control.py 4`.

## Compute accounting
One thread,2GB, OS soft105/hard110 CPU per process, shared primary lock. Direct per-copy native numeric costs use H4 sampler111; historical legacy72 retained separately. Integer work separately exposed. No transferred multiplier. Named math/library operations do not imply compiled/hardware cost; memory, control and import exclusions inherited explicitly.

## Claims
[measured] Prior k29 checkpoints/truths exist. No new performance claim.

## Alternatives not ruled out
Apparent prior wins may depend on undercounted costs; direct native tracing distinguishes this. Training level may change approximation quality; fixed recipes and all saved noisy models distinguish this without tuning. Truth errors or provenance drift could mimic changes; full hash regeneration and prior independent truth audit are required.

## Failure modes
Positive-volume tie defects remain scoped in T113; generic finite-copy success is not a theorem. Cap/parity/source failures remain unresolved and preserved. Statistical intervals conditional on fixed truth estimates do not certify exact real-network probabilities or the full ARC challenge.

Full scope/continuation claim b93dc799; FULL_PROTOCOL.md freezes all5120 noisy plus5 clean original cases. Research projection~40CPU-minutes, each shard≤110CPU. All clean outputs/parity saved, including r112 seed3 error−.0003392190026937. No recipe selection or new truth.

Incremental independent review924177cc passes all2048 seed0/1 models and4096 native site ledgers, exact per-copy budgets and bootstrap. Seed1 r85b has4 named111 individual losses (historical72 has1); no remaining-seed inference. Full panel continues unchanged.

Independent seed2 review815c5576 passes all1024 models/2048 ledgers, exact own-budget ratios and bootstrap; no individual losses in that seed. Seeds3/4 and clean/global review remain pending.

Independent seed3 reviewbaadda44 passes all1024 models/2048 ledgers and exact budget/interval contractions; both recipes have zero individual losses in this seed. Final seed4 and clean/global review remain.
