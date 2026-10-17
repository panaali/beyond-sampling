# T121 bounded cost-audit handoff

Status: implementation and frozen controls complete; final T55 review pending. Claim1e0b205 preceded implementation. DRI registers claims and owns any historical/shared-code corrections.

## Result
The exact slow R4-fixed LOOK3 implementation used by T101/T114 retains every tested value bit-for-bit, as well as n/x, segments and traces, on two analytic controls plus eight frozen clean/noisy trained cases. Its historical count charges dense Wh@Ag/Wo@Ag as though active-gate/truncated-column sparsity were implemented. Dense scalar arithmetic and an independently executed scalar sampler expose a second issue: the historical72 sampling denominator omitted comparisons/mixed units; the declared scalar sampler costs112.

The initial dense-only replacement remains a partial hybrid count and must not be used as a full result. The subsequent native-expression ledger independently counts the pinned functions, including _Sint; its matrix totals agree exactly with the first ledger. Under the declared **explicit-source-expression** convention, the eight trained sampling-equivalent costs are0.805806–0.858590 times historical equivalents; analytic zero/positive controls give1.977350/1.265606. This does not establish any accuracy/MSE result. Changing a numerator without the comparator would have given a misleading conclusion.

## What the convention does and does not establish
Count every explicit numeric arithmetic/comparison expression, including explicit integer index/control expressions; reductions cost n−1 primitive operations; special functions cost one call; sorting charges actual key comparisons. Identical AST instrumentation of the ordinary scalar sampling forward produces111 floating operations/comparisons plus1 integer correctness comparison=112 on all10 models. The floating-only subtotal excludes algorithmic class decisions and is NOT the common-unit result.

This is code-form-dependent abstract accounting. Implicit loop iteration/branch machinery, memory allocation/copies/indexing/filtering/selection, dictionary/cache lookup, casts and interpreter overhead are free on both sides. It is neither machine-instruction counting nor a complete implementation-cost certificate. All model-dependent State/row construction and in-call setup/cache misses are included; only module-level model-independent FAMS initialization is exempt. No credited sparse rescue was implemented; fast T95 remains unreviewed.

## Verification and independence
Primitive controls: zero rows_at350=291 dense multiply/add+36 masks+8 input additions+12 gate comparisons+3 Boolean ORs; zero State647=350+180 R projections+45 L projections+72 abs/comparison/negation/division; nonzero _Sint20 and zero-radius_Sint2. Chained comparisons short-circuit without evaluating a raising later operand; in-place assignment retains alias identity; sorting reproduces exact ordering and four comparisons on a three-key control. Every frozen estimate, n/x and topology counter matches baseline, and every matrix total matches the separate phase1 inventory.

T55 first independent review2c64019 checks all10 named ledgers and sampler112. Final native-expression scope/source review is pending. Independent reviewer correctly flagged that algorithmic integer comparisons cannot be dropped; same-AST sampler and inclusive explicit-expression convention resolve that mismatch, with source-form caveat retained.

## Finite alternatives checklist
- Tested: full dense dimensions instead of hypothetical active sparse execution; scalar multiply/add instead of hybrid MAC+scalar accounting; class comparisons on both estimator and sampler; same pinned source/cache retention; zero and positive controls; four fixed trained checkpoints and their predeclared noisy copies; identical source-expression instrumentation on both algorithms.
- Not tested: an actual sparse implementation, T95 fast path, hardware timing/FMA conventions, higher H/T, broader noisy MSE panels, or any new accuracy estimator. These remain distinct open possibilities and are not ruled out.
- No broad negative or dead-method verdict. The historical numerical answers survive this audit; the arithmetic convention and implementation attribution require qualification.

## Reproduce
From this folder, using the project venv and one BLAS thread: `python primitives.py`, `python panel.py`, `python arithmetic_primitives.py`, `python arithmetic_panel.py`, `python sampler_native.py`. Use the shared review lock for primitives and primary compute lock for panels,2GB limits and110CPU cap. Frozen checkpoints reside in `/workspace/work/t114_checkpoints`; every file hash and source hash is recorded. Full ledgers are JSON artifacts, not inferred from rounded summary ratios.

Research commits:856653c,30e0a23,073e748,4a76e0f,64e5809, followed by this handoff/control checkpoint. Historical outputs are preserved, not overwritten.
