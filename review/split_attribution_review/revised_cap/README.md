# T198 revised T195 cap review

## Question
Does author `2f097739` establish an outward cap and a certified pruning replay after the original narrow-cone defect?

## Method
The TASKS-only renewed claim is `886859a0`, published through `dffdffb0`, under the DRI's 02:10 assignment. `manifest.json` pins 73 files from the author tree, including the changed cap, its printed controls, 51 replay JSON files, reference inputs and numerical dependencies. Review is source/provenance only so far: no author import, cap, geometry, mass, model, random input or estimator call.

## Status
Source findings frozen; independent source/theorem review and a separately gated saved arithmetic audit remain pending. The revised source removes the original cancellation mechanism, but the available evidence does not yet establish a general floating-point outward certificate. This is not a trained-method failure.

## Results
See `SOURCE_REVIEW.md`. The archived replay covers T3 only: five seeds, two tests and five nonzero thresholds, plus one zero-threshold control. The old T4 result was removed; no replacement T4/T5 replay is present in the pinned tree. No rendered `prune_report.md` is present. A queued shell script is not completed evidence.

## Compute accounting
Only Git reads, source inspection, file copying and hashing have occurred. Zero numerical candidate or reference evaluations. The author's printed 834 checks and saved replay times are author records, not independent executions or complete resource accounting.

## Claims
- [inferred] The real-arithmetic circular-cap formula is valid under unit-ray, positive-axis and represented-domain assumptions.
- [inferred] Fixed padding constants alone do not demonstrate floating-point outwardness; their derivation and accumulation/error scope are missing.
- [inferred] A zero-axis input is not explicitly guarded. The unrestricted hull counterexample in the source note is outside the stated pointed-cell contract and does not refute the actual trained traversal.
- [measured] The pinned tree contains 51 T3 replay files and no rendered pruning table or T4/T5 replay files.

## Alternatives not ruled out
A derived floating-point enclosure on the actual finite dimension/ray-count domain, an explicit safe fallback, or the reviewed exact rational T200 cap can close the cap gap. T209 shows that determinant information can substantially tighten some bounds, but its small fixed corpus is not a pruning-budget proof. Missing T4/T5 reruns may arrive later and must be separately pinned. No method is retired by this review.

## Failure modes
Do not treat finite random checks as a universal certificate, a same-library numerical mass as exact truth, or a sum of pruning bounds as total estimator error. The underlying recipe error, wrong-centroid numerical masses, represented-domain coverage and floating accumulation remain separate. Do not silently mix the historical `N` field with the corrected report's T196 sampling charge.
