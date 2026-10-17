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
Running frozen primitive/source audit, then same-panel native cost inventory. One thread,2GB; <=20CPU primitive and<=110CPU primary-lock cycles. Independent T55 source/count review required before new cost claims.

## Results
Pending. No higher-rung or full-challenge claim.

## Compute accounting
Native explicit expressions and declared reductions/special/sort calls are counted, including all model-dependent setup. Historical Ops.n/x are preserved separately. Comparisons are charged on both estimator and sampler. Validation references/instrumentation/compilation/I/O are separately reported. No cost credit for merely hypothetical sparse execution. Source-form and memory/control exclusions follow T121 and remain prominent.

## Claims
[inferred] The fast implementation needs its own audit; slow-path accounting cannot certify it. [speculative] The true rank-one/pruning savings may survive additional scalar/setup charges. No new confirmation yet.

## Alternatives not ruled out
Unreviewed floating degeneracies/coincident-gate tolerances, rounding-sensitive finite stopping, different LOOK recipes, actual sparse rewrites, different cost conventions, higher H/T and hardware runtime. Existing limited scale failure does not retire geometry or the whole fibre family.

## Failure modes
Instrumented arithmetic changes evaluation order; dynamic cache state leaks between runs; hidden stale dependency imports; a changed approximate partition is mislabeled an exact rewrite; selection/validation is charged as estimator work or omitted setup is silently free; analytic fixture success is generalized to trained/noisy models.
