# T201 mechanism-claims review

## Question
What evidence supports the216-state/separable reconstruction claims, error
bucket breakdown and8/9 blind-region heuristic?

## Method
Initial claimcf62f4ab permits source/saved inventory and mathematical derivation
only. `frozen/` and `pins.json` preserve thirteen source/narrative artifacts.
SOURCE_INVENTORY.md distinguishes task-state sufficiency from network
reconstruction and records missing fit/cross-tab provenance. BLIND_ARGUMENT.md
derives the exact sign-region probability and the additional assumptions needed
for a rank-blind error calculation. No author source imports or model calls.

## Status
Initial bounded source/derivation evidence independently reviewed: theorem
dd64fb85 and metadata1d352ce1, both published. See HANDOFF.md.
Historical fit drivers/raw data remain unrecovered in the stated search scope.
The fixed reconstruction experiment has now completed under separately claimed
and independently source-gated scope4226647e. Raw cd272a57/shared20a452be keeps
all inputs, hidden/logit arrays and fitted coefficients. Independent saved
audit908f2a56 passes; see rebuild/FULLFIT_RESULTS.md and CHECKPOINT.md.

EXACT_STATE_LIMIT.md additionally reuses independently reviewed T138 evidence
under scopef7509964: varying omitted input1 around the old strict crossing
preserves (state,m1,m2,x9) but changes the unique network winner. This is a
theoretical implication of pinned existing certificates, independently reviewed
in dd64fb85, with no new numerical evaluation. Approximate lookup quality remains
open; no probability or practical error lower bound is claimed.

## Results and claims
[inferred, independently reviewed]216=3*9*8 states determine the true answer exactly,
which alone does not establish a faithful network model. [source observation]
state.py and separable.py implement helpers without a frozen experiment split.
[inferred, independently reviewed]11/1024 is exact;8/9 is not implied by architecture,
and even a fully rank-blind idealization has different zero/one-positive rates.
[source observation]the later ten-probe routine covers K=1 only, not the exact
blind conditional law. The new frozen reconstruction has preliminary heldout lookup task scores near
98%, but only about95.3% agreement with the original network; independent saved
review908f2a56 passes. No rung pass or deductive accuracy-estimator claim.

## Compute accounting
Initial source phase used byte/hash/git searches and written algebra only.
Subsequent independently gated synthetic controls,1024-input wiring pilot and
fixed196608-input reconstruction have separate full ledgers in HANDOFF.md and
rebuild/FULLFIT_RESULTS.md. Fullfit added195584 input vectors,191 forwardchunks
and217OLS calls, reusing the pilot once;7.467979396 scriptCPU. No fresh reference,
quadrature or native-cost calls. The initial zero-call statement is not a total
for later phases.
Source imports are deliberately avoided because common.py loads a model and
arc_eval/estimate generate400,000 reference inputs at import.

## Alternatives not ruled out
Historical driver/data may survive in another archive; results may be valid
held-out observations. Weight-specific blind-region structure may improve or
worsen the rank-blind heuristic. A freshly rebuilt lookup can generalize well
without being an exact model of hidden state or errors.

## Failure modes
Treating missing provenance as evidence of leakage; confusing target prediction
with model reconstruction; inferring independence from ReLU; summing overlapping
error buckets; using rounded prose values as exact empirical counts. Shared
FINDINGS/RESULTS corrections are proposed only; Claude DRI retains adoption.


**Snapshot scope (2026-10-20):** Some documentation snapshots are now ARC-only editorial extracts. Original manifest hashes and saved reviews describe the historical originals. See [snapshot scope and replay guidance](../../docs/SNAPSHOT_SCOPE.md) before rerunning a historical check.
