# T201 checkpoint pilot independent review

## Question
Does the frozen1024-input pilot test checkpoint wiring without fitting or changing inputs after outcomes?

## Method
Explicit delegation0ab7d45c. Source-only review of pilot, original architecture/loader, already-reviewed core and synthetic artifacts; all12 pins verified by byte hashing. No checkpoint load, model/library execution or random draws by reviewer. `source_gate.json` records exact scope.

## Status
Source gate passes after pre-run durability repair2680915d. Independent saved pilot audit remains pending actual execution/publication.

## Results
The ten-step ReLU recurrence, input/weight orientation, single library hidden hook, first-index readout and fixed parity thresholds match. Full CPU RNG states now persist before construction/assignment equality guards. Exactly1024 train-prefix inputs, no fits or holdouts are cleared.

## Compute accounting
Read-only source/AST inspection and12 file hashes; zero checkpoint loads/forwards/draws. Author90CPU soft/110CPU external primary lock, one thread,2GiB RSS assertion and distinct2500000KiB virtual cap. No full-cost claim.

## Claims
[inferred] Source supports a finite wiring discriminator; a passing result does not validate historical98%/90–91% rates or the proposed surrogate.

## Alternatives not ruled out
Floating accumulation differences and argmax flips remain observable. Summary non-sufficiency, rare/deficient training states, fitting variability and heldout behavior remain separate.

## Failure modes
No outward floating certificate, runtime/cost comparison or challenge-rung approval. Full continuation and prefix reuse require their own source and saved-output gates.
