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
Initial bounded source/derivation evidence prepared; independent review pending.
Historical fit drivers/raw data remain unrecovered in the stated search scope.
The full T201 task is not complete: any rebuilt held-out fits or new cross-tab
require a separately published fixed protocol and source/resource gate.

EXACT_STATE_LIMIT.md additionally reuses independently reviewed T138 evidence
under scopef7509964: varying omitted input1 around the old strict crossing
preserves (state,m1,m2,x9) but changes the unique network winner. This is a
theoretical implication of pinned existing certificates, pending independent
review, with no new numerical evaluation. Approximate lookup quality remains
open; no probability or practical error lower bound is claimed.

## Results and claims
[inferred, review pending]216=3*9*8 states determine the true answer exactly,
which alone does not establish a faithful network model. [source observation]
state.py and separable.py implement helpers without a frozen experiment split.
[inferred, review pending]11/1024 is exact;8/9 is not implied by architecture,
and even a fully rank-blind idealization has different zero/one-positive rates.
[source observation]the later ten-probe routine covers K=1 only, not the exact
blind conditional law. No new accuracy result, significance test or rung pass.

## Compute accounting
Byte/hash/git searches and exact written algebra only; model forwards, fitting,
new Gaussian inputs, quadrature, reference and native-cost calls all zero.
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
