# T283 independent saved arithmetic result

## Question
Do the retained T271/T277 rate summaries and the producer's scalar audit reproduce under independent arithmetic, and can the four prefactor mismatches be explained by the predeclared denominator diagnostic?

## Method
Frozen c77007fa/shared abc4f944; independent78 source gate4fa2c650/shared8f6e1e7c; immutable execution gate409fadaa/sharedafc15297 published before the sole run. Exact common-lattice integer moments, Decimal80 uncentered OLS and integer-degree Student cosine-integral recurrence independently check the saved numbers. Source and diagnostics were fixed before decoding. See PROTOCOL.md, check.py and output/results.json. No producer/scientific-library/model/array/RNG/MC replay.

## Status
Complete/pass_all. One successful scientific attempt; no arithmetic failure or retry. The earlier source-review metadata ancestry stop is separately preserved in ../metadata_preflight.json and resolved by a zero-tree bridge before execution. Source/gate/input bytes remain immutable.

## Results
[measured] All289 T271 rows/22 groups and720 T277 rows/90 cells/18 models satisfy fixed coverage and arithmetic checks. All1,186 producer comparison records, original fit fields, reconstructed summaries, cost/error conventions, positivity selection and finite rule pass within the frozen allowances. All four own progress events completed.

[measured] T271's retained regression CI contains0.5 for the target, s0.005_0 and s0.01_1, but not s0.005_1 or s0.01_0. Thus the blanket statement that every model's interval contains0.5 is false even under its historical regression formula. Inclusion itself is not equivalence.

[measured] T277 reproduces rho=0.02857142857 and s=0.32499950843, CI approximately[0.04511388109,0.60488513577]. The specified finite rule returns refuted through rho>−0.2; the separate all-alpha-within0.5±0.15 condition is false. This is the rule's scoped finite outcome, not a broad dimension/rate conclusion.

[measured] All four original prefactor mismatches remain mismatches under fit.py's MC-accuracy denominator. Each matches the same length/budget median when using the retained exact-summary accuracy denominator at T3/T4: T3@16k, T3@256k, T4@16k and T4@256k. The other eight numeric leaves match the MC convention and the unchanged diagnostic convention. No values, tolerance or original records were replaced.

[inferred] The four discrepancies are compatible with a denominator convention difference. Because the separate prefactor generating script is absent, this does not prove how that file was generated, authenticate its history or reconstruct missing source semantics. Both original and alternative tables and all leaf matches are retained in output/results.json.

## Compute accounting
One pinned /usr/bin/python3 stdlib review process:0.391379808 internal CPU seconds,24,368KiB maximum RSS. Actual limits20/20 CPU and2GiB/2GiB. Bash timing separately0.44real/0.39user/0.05sys. No candidate calls, new truths or sample-count reconstruction. Execution command and caps are in PROTOCOL.md; run1.log is preserved. No rerun is needed to consume these results.

## Claims
[inferred] This pass establishes saved scalar consistency at the fixed tolerances and the listed descriptive mismatches. It does not validate nominal regression-CI coverage under shared scrambles/references, asymptotic rates, dimension independence, equivalence to0.5, or prefactor as the sole explanation. Retained exact-summary references and cost units remain inherited. T277 enum aggregate has no generating rows and remains provenance-only. No current HP2 score, cost-floor, sealed-input access or leaderboard adoption claim.

## Alternatives not ruled out
Reference uncertainty and cross-budget dependence can alter the historical fits. Budget-dependent allocation can change finite slopes. A different retained denominator explains the four arithmetic values without establishing their generating implementation. Longer ranges or different models could behave differently.

## Failure modes
Allowances are saved-display consistency tolerances, not numerical enclosures. Missing raw trials, generating scripts or runtime/model bindings remain missing. Outputs and all failures are immutable; subsequent numerical expansions or changed tolerances require their own scope and gate.
