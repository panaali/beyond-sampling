# T461: independent saved-evidence audit of T456

## Question
Does the published IS-v4 panel support its reported score, selection chronology and interpretation?

## Method
`check.py` reads immutable Git blobs at `a52da0d6a57d29aa2f76daa74c3360222f33aec2` using only the Python standard library. It independently reconstructs the 15 panel identities, estimates/reference arithmetic, gate predicates, z scores, per-point and aggregate F, comparison counts, and committed freeze order. It does not import or run producer code. File hashes and all outputs are in `checks.json`.

Reproduce from a repository containing the producer history:
```
python review/is_v4_saved_audit/check.py
```

## Status
Saved arithmetic passes. This is not an independent estimator implementation, repeated-run calibration test, or full cost audit. T456 remains scientifically unverified beyond these checks.

## Results
- [measured from saved records] All 15 panel identities match the preregistered name/threshold cycle. All estimates are byte-identical at freeze `91e85991` and final publication; no `truth_*.json` exists in that folder at freeze. Panel and estimator/runner code are unchanged from preregistration `5ae28296`. This establishes committed chronology, not a proof that nobody privately viewed truth.
- [measured from saved records] F = 0.0638351278350 in the author's network-pass convention; 0.272390707795 when draws are charged. Both point scores satisfy strict F <= 1 and lenient F <= 10. All 15 reported combined-error z scores are within 2; maximum 1.46103; no gate fires. Prefilter comparisons reproduce 6/15 and 13/15.
- **Reporting correction:** distinct-triple cardinality increases on **14/15**, not the README/INDEX's 13/15. Only index 3 has unchanged cardinality, but it replaces three triples. Set differences show at least one new triple on **15/15** points. Neither observation proves discovery causes accuracy improvement.
- **Reporting precision:** ratios range **0.9677683–1.0311090**, so say “about 3.3%” or “rounded ratios 0.97–1.03,” not literally “within 3%.”
- [measured arithmetic diagnostic] Mean reported reference variance is **76.53% of the observed mean squared estimate/reference discrepancy**. Raw F includes noisy references; it is not an independently measured noise-free estimator MSE. We do not replace it with a post-hoc corrected score.

## Compute accounting
The audit performs only arithmetic over 31 published small JSON records plus source/history checks; zero model calls, random draws, geometry calls, or new truths. Producer C includes network passes and descent time converted at 4 microseconds/FPE; C_draws additionally prices each draw as one pass. `is4.py` does not instrument mixture-density evaluation, component fitting/eigendecompositions, and all proposal processing as T196 FLOPs. Neither score convention is a demonstrated full-operation count; this audit reproduces their stated conventions only. The full weight-dependent cost audit is a separate follow-up.

## Claims and interpretation
- [inferred] The observed panel passes its preregistered consistency criterion. Fifteen one-shot z comparisons do not establish repeated-run interval coverage or an honest-interval guarantee.
- [inferred] The heavy-tail gate remains unvalidated: it never activates here. Source/analytic adversarial fixtures can cheaply test branches before expensive fresh-panel work.
- [inferred] “Fresh” means unused by T403/T453/T454 and selected by a model-free rule; the author discloses earlier rare-event use of some DEV3 models. Do not describe this as a wholly unseen sealed model panel.
- [inferred] Both estimates and truth use the same defensive sampler implementation. Independent seeds do not test common sampler/model-loading bugs. A different implementation is needed for scientific replication.

## Alternatives not ruled out
The observed gain may depend on easier panel composition; missing modes may evade the silent gate; noisy reference values can move the empirical F; omitted weight-dependent processing can raise full cost. Checks here rule out arithmetic/transcription and committed-freeze inconsistencies, not these explanations. No extra panel, tuning, or author rerun was performed.

## Failure modes
The truth runner stops after a hit floor in blocks and uses a Poisson-style SE; this is not an anytime-valid confidence construction. The estimator can extend after inspecting relative SE, although no retained panel row extended. Neither general interval calibration nor optional-stopping validity follows from this finite pass. Raw sample weights and component-level timing are not in the audited summary records, so their SE/cost inputs cannot be independently reconstructed here.

Independent final check: [T459 reviewer](../dri_takeover_readiness/T461_FINAL_REVIEW.md).
