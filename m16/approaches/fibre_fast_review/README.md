# T128 frozen fast-fibre independent audit

## Question
What does the actual T95 fast F2L2 implementation cost under the same explicit-source-expression convention as T121, and are its range/rank-one/cache primitives consistent with their claimed invariants? Claimbbc6a3e was pushed before implementation. This folder owns local snapshots only; Claude retains author code and T4 work.

## Method
Pin fibre_fast.py, fibre.py, fiber3.py and verify_fast.py atbbc6a3e, recording byte hashes. Default LOOK2 and all six optimization flags stay on. Reuse the ten frozen T121 models (two synthetic, four trained checkpoints each clean plus fixed absolute-.05 PCG64seed124 copy). Add two constant-label controls on the positive synthetic hidden dynamics: allzero readout and duplicate positive rows0/1 with zero row2; both have exact accuracy1/3 and stable first-index ties. No new checkpoint/noise sweep.

First freeze primitive checks: _range on25 coefficient pairs from{-2,-1,0,1,2}² over eight fixed intervals (including wrap, extrema, narrow and endpoint cases), independently enumerate derivative roots; Piece._flip on all64 gate subsets of one fixed dyadic H2T3 model, comparing the resulting rows/logits against a separate direct prescribed-gate recurrence, including projections and candidate reuse. Record all discrepancies, not just maxima. These test real range and algebra, not arbitrary floating-point topology. Bounds/tolerances and costs must be explicit.

Then independently inventory native operations while retaining original operands/order and baseline outputs/counters. T121's wrapper needs explicit support for fast-only outer products, array min/max/hypot/equality, integer division/divmod and builtin sum. Validate each new primitive before interpreting the fixed panel. Every call's integer class/index expressions are included, implicit interpreter/loop/branch/memory machinery is free, and a matched same-AST sampler includes correctness. This is a code-form-dependent abstract convention, not hardware cycles or complete implementation cost. All model-dependent Piece construction, incremental updates, state checks, range caches and boundary setup count. Data-independent FAMS initialization is exempt.

Competing explanations: sparse/rank-one work genuinely lowers arithmetic; inherited costs credit dense operations as sparse or omit range/filter/state checks; instrumentation changes a branch or cache lifetime; known approximate stopping/partition sensitivity changes slow-versus-fast accuracy without a primitive defect. Primitive invariants and unchanged-native parity separate these. Only after those pass may independent analytic/exact truth compare the fixed new models; no inference from slow-fast equality alone.

## Prior evidence not repeated
T99 already checked471 frozen T95 models, documented LOOK2 scale/partition sensitivity, verified LOOK64 reached every clamp on30 models, and excluded the older id(st)-reuse defect from this Piece cache design. Its finite omission/cap panels and failed canonical normalization remain valid scoped evidence. Do not rerun that sweep or rebrand the known mismatch as a new bug. The author README explicitly admits approximate slow-fast differences despite an older exact-reduction header. T121 audits slow R4 LOOK3, not this fast LOOK2 recipe.

## Status
Done: bounded primitive/source audit, native inventory and fixed-panel references, independently reviewed39f7deb. One thread,2GB; <=20CPU primitive and<=110CPU primary-lock cycles. Independent T55 source/count review required before new cost claims.

## Results
All12 baseline values, current counters, segments, traces, constraint counts and per-family error hex values match after native instrumentation. All200 range controls agree within4.44e-16; all64 fixed-gate flip/projection/cache controls agree exactly. Three constant-label models have accuracy1/3. Same-AST sampling count112 holds throughout. New primitive counts and raw totals/ratios were independently checked. No higher-rung or full-challenge claim.

## Compute accounting
Native explicit expressions and declared reductions/special/sort calls are counted, including all model-dependent setup. Historical Ops.n/x are preserved separately. Comparisons are charged on both estimator and sampler. Validation references/instrumentation/compilation/I/O are separately reported. No cost credit for merely hypothetical sparse execution. Source-form and memory/control exclusions follow T121 and remain prominent.

## Claims
[inferred] The fast implementation needs its own audit; slow-path accounting cannot certify it. [speculative] The true rank-one/pruning savings may survive additional scalar/setup charges. No new confirmation yet.

## Alternatives not ruled out
Unreviewed floating degeneracies/coincident-gate tolerances, rounding-sensitive finite stopping, different LOOK recipes, actual sparse rewrites, different cost conventions, higher H/T and hardware runtime. Existing limited scale failure does not retire geometry or the whole fibre family.

## Failure modes
Instrumented arithmetic changes evaluation order; dynamic cache state leaks between runs; hidden stale dependency imports; a changed approximate partition is mislabeled an exact rewrite; selection/validation is charged as estimator work or omitted setup is silently free; analytic fixture success is generalized to trained/noisy models.

## Frozen counter-provenance discriminator
The first panel records currentbbc6a3e dependencies. Its imported fibre.py has adopted T97's X additions directly into Ops.n; fibre_fast.py and fiber3.py remain byte-identical to T99's older sources. Thus current seed0 n347954 is not old T95/T99 n211218. Before using a historical comparison, pin fibre_authorcount.py and prove via AST that every imported helper and transitive helper differs only by diagnostic X calls. Record X charges in unchanged execution; subtract them to reconstruct the older counter, retaining current n and the independent native ledger. Check every estimate/topology count against the frozen panel and the known original seed0 counter. No arithmetic or output change, no new model selection. Initial panel field names containing historical mean current pinned counter only; original reported-era values will be in counter_provenance_results.json.

## Frozen numerical-reference check
After all twelve native parity checks and primitive reviews passed, run T114's independently authored ordered-rank-chart spherical polygon reference on the same12 frozen models, with analytic1/3 checks on the three constant-label cases. Pin its source hash. This is validation work, not estimator work, and does not repeat the old T99 panel. Use all twelve outputs, preserving numerical clipping/roundoff qualifications. Compute conditional squared-bias divided by Bernoulli sampling variance at the declared expression budget, retaining zero/positive controls. This finite panel is not a full noisy ARC protocol pass or a rigorous truth enclosure; independently review all ratios before reporting.

## Frozen cap check
The fast source retains two silent64-iteration walk caps. Before interpreting the fixed-panel accuracy comparison, instrument only these range iterators: preserve identical yielded integers and record whether a loop requests iteration65 (normal exhaustion) or exits by break/return. Check unchanged values/counters/topology on the same12 cases, recording maxima and all exhaustions. Do not extend LOOK or enlarge caps; this is a new source-safety check on the fixed current panel, not repetition of T99's older LOOK64 experiment. Tiny-angle skips and floating predicate errors remain separate qualifications even if neither cap is hit.

## Bounded results and independent review
[measured] Trained explicit-expression sample-equivalent costs are1.872486–1.887998 times the original T95/T99 author-counter equivalents, or1.108565–1.151513 times the current helper-adjusted equivalents. These are distinct source-era comparisons: fast/fiber3 hashes are unchanged, but current fibre.py folds X into n. All four imported/transitive helper ASTs match the original after removing only diagnostic X calls; current n=original n+recorded X, with original seed0 n211218 reproduced. The first panel's fields named historical refer only to its pinned current counter; counter_provenance_results.json gives actual original-era numbers.

[measured] Independent ordered-chart polygon integration on the same12 cases gives worst conditional squared-bias/Bernoulli-variance ratio0.1923764202907, on H4T3 seed1 clean. Next are0.0848432790787 (k24seed4 clean) and0.0536124835498 (its fixed noisy copy). The remaining cases are smaller, all below1 under the declared explicit-expression convention. These are fixed-case numerical comparisons, not a full noisy protocol pass or a certified truth enclosure. The positive synthetic truth0.327400736766831 agrees with fast LOOK2; the older slow LOOK3 value0.337182743105324 is retained as a distinct approximate recipe result, not silently substituted.

[measured] Neither64-step cap is exhausted in any frozen case; maximum forward/reverse iterations9/4. The cap diagnostic preserves values/counters/topology. It does not cover all models, tiny-angle skipped intervals or general floating topology.

Independent T55 review39f7deb recomputes all12 site totals, both source-era budgets, every conditional squared-bias ratio, helper AST provenance, and the fixed-gate/range algebra. Cap iterator semantics separately source-reviewed by T55; written appendix pending. Total phase CPU is recorded in each JSON (native panel7.62154CPU with imports, references1.01503CPU including model loading, provenance1.98CPU, cap check2.02CPU, primitives0.04CPU plus inventory controls). These are diagnostic costs, not the estimator's challenge timing.

## Reproduce and remaining alternatives
In this folder, project venv/one BLAS thread: `python primitives.py`, `python inventory_primitives.py`, `python panel.py`, `python counter_provenance.py`, `python references.py`, `python caps.py`. Use review lock for bounded primitive/reference work and primary lock for the panel,2GB limits. Current/source-era hashes live in snapshot/manifest.json; checkpoint hashes and every noisy model's fixed construction are recorded.

Tested alternatives: hypothetical sparse charges versus actual dense/rank-one work; original versus adopted helper counts; integer comparisons on both sides; independent range extrema and fixed-gate recurrence; analytic ties; numerical reference; silent walk caps on the frozen cases. Untested: full original471-case cost recount, larger H/T, hardware/FMA/implicit-operation conventions, tolerance-proof or interval-enclosed geometry, new sparse implementation, all possible floating degeneracies and global cache termination. No method-family negative or complete hardware-cost certificate follows. Full existing-suite recount is a distinct justified follow-up because T99's reported ratios used the old counter and source-dependent extra costs must be checked across its panel.
