# Independent T234 capacity-study contract (T237)

## Question
What can the proposed longer/enriched training experiment establish about the
H16T10 network, and what source/provenance is needed to distinguish its effects?

## Method
Source-only claime823d1d3. `manifest.json` pins the owner directive, freshly
fetched author-branch inventory and local alg-zoo source snapshots.
`SOURCE_REVIEW.md` describes the LR-horizon coupling, weights-only checkpoint
helper and default validation size. `STATISTICAL_CONTRACT.md` separates feasible
accuracy from architecture ceilings, MC from seed variation, selection/stopping
and training-law versus evaluation-law effects. Owner training remains unchanged.

## Status
Written source/theorem phase only; independent review pending with fundamentals.
No published capacity-study driver was present at frozen author headf02482d6.
No inference is made about unpublished author work or future Colab execution.

## Results
[inferred from pinned source] Changing default n_train also changes the LR decay
horizon. Existing checkpoint saving omits optimizer/scheduler/RNG state. Default
validation2^24 is below the requested10^8 final evaluation. These are attribution
requirements, not detected defects in an unpublished new driver.
[inferred, review pending] Finite trained models can lower-bound feasible
architecture accuracy; finite optimizer failure cannot upper-bound the architecture
supremum. No new training outcome or accuracy measurement is reported.

## Compute accounting
Only text/hash/Git inventory and documentation. Zero imports of training/model
code, checkpoint decoding, raw-array arithmetic, training, model/MC/RNG, GPU or
Colab calls. No owner compute budget is consumed or modified by this review.

## Claims
Source facts are conditional on the pinned local code, not proven executed
Colab source. Written inference requires independent review. All owner reference
accuracies remain reported/unverified here. No capacity ceiling, convergence,
training improvement, ARC solution or universal negative is claimed.

## Alternatives not ruled out
Other optimizers, initializations, objectives, schedules, representations and
larger budgets can change reachable performance. A finite plateau may be
optimization-limited, data-limited or architecture-related; this study alone
need not distinguish all three. A separate representation proof could do more.

## Failure modes
Schedule/data confounding, weights-only restarts, selected-best/optional-stopping
bias, reused evaluation streams, missing seed failures, enriched-distribution
misreporting and overlapping anatomy bins. Missing provenance limits attribution;
it does not negate valid measured results or authorize new numerical work here.
