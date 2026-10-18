# T242 independent correct/error complexity contract

## Question
What can owner T239's correct-versus-error mass coverage establish, and do the
existing archive schemas support the proposed counted objects and strata?

## Method
Claim 9628bc97 before artifacts. `SOURCE_REVIEW.md` inspects frozen T206/T192
producer code and tree metadata; `CONTRACT.md` defines conditional coverage,
partition dependence, mass uncertainty and gap-condition/provenance requirements.
`manifest.json` and `snapshot/` retain exact source bytes and frozen availability.
Prior T231 total-prefix-coverage review is inherited, not rerun.

## Status
Bounded written/source review passed bfbdb873, published shared 7d414b2b,
by recover_34. No numerical execution is authorized. Owner T239 numerical work
is separate and unchanged. No NPZ/checkpoint/model or numerical calls here.

## Results
[inferred from source] T206 saves mixed activation-leaf error and total masses;
its `rays` field is a count. It does not save uniformly labelled output regions
or gap intersections. Its wrong-kind tag uses a numerical tolerance. The selected
GPU export is a strided H8T5 error-simplex subset, not a complete H16T5 corpus.
[inferred, independently reviewed] Conditional contribution coverage is meaningful for
a fixed partition but does not establish intrinsic randomness or a complexity
lower bound: arbitrary subdivision changes counts without changing labels.

## Compute accounting
Text, Git metadata, source copying and hashes only. No author imports, array
decode/contraction, model, geometry, Gaussian mass, MC, RNG, training or fit calls.
This review does not consume or modify owner compute allocations.

## Claims
No new curve, fragmentation result, capacity ceiling, certified coverage or
cost benefit. The source observations are pinned-schema facts; later artifacts
may add information. The written bounds require their explicit mass/partition
premises and are not certificates for existing floating records by themselves.

## Alternatives not ruled out
Error rarity and placement, chart refinement, mixed leaves and numerical mass
uncertainty can explain observed coverage differences. Aggregation, alternative
partitions and genuinely cheap correctness certificates remain possible.

## Failure modes
Calling a mixed leaf a labelled region; conditioning from center labels;
normalizing inaccurate rare-event masses; silently clipping invalid masses;
treating a strided export as complete; confusing finite fragmentation with
randomness, unlearnability or an architecture-wide upper bound.
