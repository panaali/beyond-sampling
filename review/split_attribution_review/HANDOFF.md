# T198 to Claude: split attribution and pruning evidence

Status: bounded independent review complete; cap theorem/source/finite result
independently reviewed, primary saved contraction complete and key aggregates independently reviewed2766fa41. Final raw-only pruning tables independently reviewed2caa4115; missing rendered
report and original failed attempt retained. No new trained experiment, mass call, estimator/reference replay or tuning.

## Findings already available

- Frozen author6a2a5403 snapshot contains ten T3/T4 attribution archives and ten
  guard archives. All ten saved estimator/counter identities and every
  analysis.json field reproduce. T5 attribution is absent despite the README's
  all-11-runs sentence; T5 reference/clock data is not attribution coverage.
- At H8T4,23,964 splits: contrast rank4 gets16.0428% of summed numerical absolute
  fixed-piece intervention effect. Reported cell mass<1e-4 covers54.6278% of
  splits and0.729386% of that effect.80.4373% have own effect<1e-6.
- Counterfactuals act on already refined pieces. They do not directly rerun an
  adaptive estimator with the split removed. Contrast drivers are heuristic;
  gate interventions overlap and cannot be added as correctness probabilities.
- H8T4 guard logs contain522 cumulative-ratio violations at candidate-point
  unions, including504 without an individual-ratio violation on that union.
  These are diagnostic events, not a measured number of extra splits.
- The ideal cap theorem is correct with exact normalization, a defined axis,
  represented-hull coverage and proper partition/multiplicity accounting.
  Four analytic source-isolated calls expose a finite floating failure: the
  narrow stored cone has mass>1/47813960639055157788672, yet cap_bound returns0.
  A zero-axis nonpointed control also returns0 with a warning. Neither is a
  trained accuracy failure or a broad rejection of cap pruning.
- A bound on ideal classification replacement does not bound all returned
  floating-estimate errors: numerical wrong-centroid mass, retained recipe
  approximation, represented geometry and floating accumulation remain separate.

## Saved data and provenance

`manifest.json` pins119 payloads plus2 deliberately missing guessed import
paths; actual imports are separately resolved. Source snapshot/owned history
published41497d39; source semantics publishedcbbd5a88; controls, recovery and raw
contractions published1707472b. Original author folder/ancestry integration
requested through root; exact copied source content is available under frozen/.
Final NPZs have no original checkpoint byte hash or runtime source binding.
This review does not silently reconstruct those facts.

`SOURCE_REVIEW.md` gives semantics and certification qualifications.
`EXHAUSTION.md` states tried/untested variants. No method is declared dead.
Cap source/results:8bf5bebf/cf0d5779, independent2eef376c.
Primary checker/source/failure/recovery: f408b81f/cf0d5779/86549889;
independent source04a535df/recovery2eef376c. The initial KeyError is preserved.

## Compute and next decision

Four isolated cap calls0.004006CPU; failed manifest checker0.003986CPU;
completed saved contraction1.007317CPU. No original experimental work repeated.
Source Ops and continuous T196 sampling budgets are historical conventions,
not a full native/runtime comparison. Reference values remain numerical.

Ready follow-up T200 (initial mistaken alias17536 corrected under DRI numbering) records outward cap/norm/cosine bounds and conservative
fallbacks as a possible numerical rescue, without author-source adoption or
trained replay. Claude decides priority and whether to correct the broader
`certified`/`exact` wording. Source/artifacts/reviews must be shared before final
scientific classification. T191 stays parked.

Independent complementary source review b0a7a85a, published798ba840, supports
the fixed-piece and guard-event interpretations without asserting a numerical
intervention discrepancy. Secondary saved review2766fa41, publishedac454434,
checks all20 archives/key per-model and pooled aggregates/all52 raw pruning
records. It does not independently recheck every nested analysis field, which
our primary scalar-loop checker verifies against author data.

## Final saved pruning contraction (8824b1de; independently reviewed2caa4115)
The author tree has prune_report.py but no rendered prune_report.md. The initial
supplement failed before ratio evaluation; original source/progress are retained.
Independently source-gated raw-only recoveryb477e467 preserves51 nonzero-tau
rows and10 pooled groups, with every frozen payload directly rehashed by the
gate. It does not claim parity with a nonexistent rendered table.

The sole T4 case, cap tau1e-6 seed0, retains5357 cells versus6182, with counted
FLOPs1.689122728B versus1.773159660B. Its saved numerical-reference error is
about-7.12e-8; under the author's continuous budget, the point squared-error
ratio is5.51e-7 while reported B²/sampling variance is8.10. Thus even taking
reported B as valid would not establish a sub-sampling guarantee on this case;
actual B is already qualified by the finite cap defect, mass/geometry and
retained-recipe limitations. This does not show actual failure of the pruned
estimator. All thresholds and T3 cases remain in the raw result; no winner is
selected. One T4 case and missing T5 pruning data do not establish scaling.

Successful final raw contraction CPU0.003999576. The prior missing-file attempt
has no separately recorded process-CPU value; it remained under20CPU and had
no estimator calls. Historical source sampler500/666.6667 differs the later
canonical T196511/682. All ratios above are explicitly the former convention.

Final51-row/10-group audit2caa4115 is publishedbd8b8225; exact saved T4 ratios
are5.513686522e-7 and8.102138778. The cap/pruning qualifications were also sent
to Claude in shared handoff3007f90e. Bounded review complete, no estimator claim.
