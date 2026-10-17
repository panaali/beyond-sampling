# T115 k29 margin continuation

## Question
Do the two frozen margin recipes remain competitive on the missing k29 training level when costs are measured directly under the reviewed named-source convention?

## Method
Claim5be203e1. Reuse five T90 checkpoints and all5120 existing absolute.05 noisy-copy truths (independent e49308c). No training, MC or new truth. Pin all bytes, regenerate every noisy model hash, and use unchanged T176/T172 tracers for r85b/r112. Two fixed controls are seed0/noise20/copy0 and seed4/noise23/copy255; require baseline/traced output hex, legacy, STATS and grouped errors parity. Then first64 seed0/noise20 copies form the cost/parity gate before expansion. All failures preserved.

## Status
Provenance and source gate in progress. No trained recipe evaluations yet.

## Results
Pending. Commands: `python provenance.py`; after independent gate `python control.py 0` and `python control.py 4`.

## Compute accounting
One thread,2GB, OS soft105/hard110 CPU per process, shared primary lock. Direct per-copy native numeric costs use H4 sampler111; historical legacy72 retained separately. Integer work separately exposed. No transferred multiplier. Named math/library operations do not imply compiled/hardware cost; memory, control and import exclusions inherited explicitly.

## Claims
[measured] Prior k29 checkpoints/truths exist. No new performance claim.

## Alternatives not ruled out
Apparent prior wins may depend on undercounted costs; direct native tracing distinguishes this. Training level may change approximation quality; fixed recipes and all saved noisy models distinguish this without tuning. Truth errors or provenance drift could mimic changes; full hash regeneration and prior independent truth audit are required.

## Failure modes
Positive-volume tie defects remain scoped in T113; generic finite-copy success is not a theorem. Cap/parity/source failures remain unresolved and preserved. Statistical intervals conditional on fixed truth estimates do not certify exact real-network probabilities or the full ARC challenge.
