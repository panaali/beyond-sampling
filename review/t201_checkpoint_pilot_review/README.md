# T201 checkpoint pilot independent review

## Question
Does the frozen1024-input pilot test checkpoint wiring without fitting or changing inputs after outcomes?

## Method
Explicit delegation0ab7d45c. Source-only review of pilot, original architecture/loader, already-reviewed core and synthetic artifacts; all12 pins verified by byte hashing. No checkpoint load, model/library execution or random draws by reviewer. `source_gate.json` records exact scope.

## Status
Source gate passes after pre-run durability repair2680915d. Completed author pilot5c742488 independently passes saved_results.json; checker2072c1f7 source-reviewed96e6c73b before sole saved run.

## Results
All8 archives and1024 observations independently pass hash/shape/dtype/conversion/state/label/gap/RNG/count checks. Maximum hidden difference4.973799150320701e-14 and logit difference2.9665159217984183e-13, with all1024 labels identical. No fits or holdouts ran. The ten-step ReLU recurrence, input/weight orientation, single library hidden hook, first-index readout and fixed parity thresholds match. Full CPU RNG states now persist before construction/assignment equality guards. Exactly1024 train-prefix inputs, no fits or holdouts are cleared.

## Compute accounting
Source inspection plus saved-only array contraction. Author.849919784CPU/206556KiB, one candidate and one library1024-sequence forward; reviewer.019671317CPU excludes NumPy import, zero checkpoint loads/forwards/fits/draws. Author90CPU soft/110CPU external primary lock, one thread,2GiB RSS assertion and distinct2500000KiB virtual cap. No full-cost claim.

## Claims
[inferred] Source supports a finite wiring discriminator; a passing result does not validate historical98%/90–91% rates or the proposed surrogate.

## Alternatives not ruled out
Floating accumulation differences and argmax flips remain observable. Summary non-sufficiency, rare/deficient training states, fitting variability and heldout behavior remain separate.

## Failure modes
No outward floating certificate, runtime/cost comparison or challenge-rung approval. Full continuation and prefix reuse require their own source and saved-output gates.

Saved audit command: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash -c 'ulimit -v 2097152; ulimit -t 20; flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python review/t201_checkpoint_pilot_review/check_saved.py'`. Completed outputs are immutable; do not rerun. Initial PCG64 state reconstructed without draws; terminal advancement not replayed. Stored float32 weight hashes are checked from exact float64 roundtrip against source-gated author records, without independently deserializing checkpoint.
