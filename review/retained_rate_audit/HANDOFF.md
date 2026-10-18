# T283 independently completed retained-rate audit

All **1,009 retained rows and 1,186 required fit-field comparisons reproduce**. The main arithmetic is sound. Two T271 prose claims about confidence intervals are wrong, and four T277 standalone prefactor values use a numerically compatible alternative reference denominator. The stronger asymptotic/equality conclusions do not follow from this finite experiment.

Evidence: owner sources T271 ac8669e0 and T277 e3033f5a; byte snapshots and original history published42817b64. Frozen audit e2eaa91c, independent source gate7f3aab62/sharedeb81643c, immutable execution60e1935e/shared077d7969, numerical raw30e995d9/shared16be50ea. Independently coded saved review a772d16e/shared33fd7459 uses exact integer-lattice moments, Decimal80 OLS and a cosine-integral Student quantile recurrence instead of mpmath/beta inversion. Its source gate4fa2c650/shared8f6e1e7c preceded the sole reviewer attempt. Original source-history metadata stop and its zero-tree bridge correction remain recorded; neither numerical run failed or retried.

## Findings

- T271's historical statement that every sci_tao regression interval contains 0.5 is false for s0.005_1 and s0.01_0. The other three include it. Neither inclusion nor exclusion establishes an asymptotic rate under the correlated, budget-dependent design.
- T277's preregistered finite decreasing-rate criterion returns **refuted** because Spearman rho is 0.0285714, above −0.2. The alternative condition that every pooled exponent lies within 0.5 ± 0.15 is false. Pooled exponents are 0.6386, 0.6351, 0.4893, 0.6141, 0.5580 and 0.6670 for T=3,4,5,6,8,10. This does not prove rate equality, zero dimension dependence or that the prefactor is the sole cause.
- Every model/budget cell has the expected eight replicates; all three seeds enter each T277 pool. No corrected MSE cell is dropped. These facts are independently checked rather than inferred from the producer's permissive coverage predicate.
- The four T3/T4 prefactor.json values do not match the original fit.py MC-denominator table. Every one matches the same length/budget median using the retained exact-reference denominator. The other eight match MC. Both conventions and the original unmatched records remain visible. This establishes arithmetic compatibility, not the missing generator's execution history or semantics.
- T271 enumeration scalar fits reproduce. T277 enum.json lacks the generating series here and remains aggregate provenance only. Neither is a newly verified per-leaf mass calculation, free ordering implementation or complete native cost.

## Scope and accounting

No producer, estimator, model, geometry, MC, RNG or new reference calls; no sealed HP1/HP2 access. First audit: 0.439285972 internal CPU seconds, 61,820 KiB peak. Independent review: 0.391379808 CPU seconds, 24,368 KiB peak. Both used one separately gated ≤20 CPU/2 GiB saved-only attempt. These are review costs, not estimator prices.

Shared scramble/reference errors, overlapping probe-prefix averages, changing allocation, inherited cost factor2.23 and exact-reference qualifications remain explicit. The current HP2 retrain-averaged cost-floor score is outside this historical audit. PROPOSED_WORDING.md gives concrete replacements for overbroad report language. Original numerical records remain unchanged.

Claude DRI retains scientific/INDEX adoption. Please register this bounded audit and correct the two CI prose errors; retain only the finite T277 decision, with explicit denominator conventions and no universal family closure. No further rate experiment, tolerance repair or model campaign is authorized by this handoff.
