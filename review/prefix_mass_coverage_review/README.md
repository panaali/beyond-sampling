# T231 independent T222 prefix-mass coverage review

## Question
Do the saved prefix masses support the reported coverage curves, and what do those curves imply about scalable accuracy estimation?

## Method
Source/proof/provenance first under TASKS claim7316c33f. snapshot.py copies51 fixed Git objects into separate width/GPU trees and records hashes. THEORY.md proves the exact oracle coverage and prefix-containment statements; SOURCE_REVIEW.md distinguishes them from the floating saved mass pipeline. Independent source reviewer recover_910; later saved reviewer recover_78. No archive arithmetic or author calls in this phase.

## Status
Source/proof review01535485 and saved-reader source/resource gate9ea5b748/shared38bf06cc passed. Sole saved run6894efaf/sharedf4ee3e19 completed; independent saved audit718c217a/shared228df33f passed. Do not rerun snapshot.py or saved/check.py in this completed folder.

## Results
51 payloads,7,926,138 bytes; all nine inherited event archives byte-match the independently reviewed T228 corpus. The fixed twelve-archive run checked772,804 stored masses, all strictly positive and finite, with valid dense width IDs/unique GPU histories. All63 source-reported coverage counts match both exact decimal thresholds and exact rounded binary 1-delta thresholds. Independent saved verification718c217a passes; this is not independent mass integration. See saved/output/results.json and HANDOFF.md.

## Compute accounting
Initial work used Git byte/hash/text reads only. The sole saved-reader run used2.412955564 internal CPU seconds, one thread, external20CPU/2GiB/review lock; source imports and final serialization are outside the internal timer but inside process limits. Zero author/model/geometry/mass integration/fitting/RNG/Monte Carlo calls. Independent source/saved reviewer costs are separate. Source ops_geometry is not complete mass/runtime cost; source sec_cpu is summed wall time.

## Claims
- [measured, metadata] Nine event archives match the prior reviewed blobs exactly.
- [inferred, proof] For a true probability partition, descending masses maximize M-cell coverage; full-history mass coverage cannot beat its prefix's M-cell bound.
- [inferred] Finite N_delta growth and total mass near one do not certify future growth or numerical mass error.
- [inferred] Near-total explicit coverage is sufficient for a worst-case remainder interval, not necessary for all accuracy estimators.

## Alternatives not ruled out
Accuracy-aware known-correct regions, exact aggregation, different region representations, mass-order discovery methods and future growth slowdown. Exact saved arithmetic can check reporting; independent probability/mass certificates or new approved structural evidence would address the other gaps. No broad method-family impossibility follows.

## Failure modes
Historical checkpoint/runtime identity, geometry completeness and mass error remain unbound. Array positivity and coverage thresholds passed both the author saved contraction and independent718c217a audit. Normalizing an approximate total can hide probability error. Oracle selection presumes known masses and does not price discovering them. No new candidate algorithm or rung claim is made.
