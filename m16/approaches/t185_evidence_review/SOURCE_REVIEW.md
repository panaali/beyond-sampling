# Source and provenance inventory

Pinned source SHA a88f62c41a94af636399675e92ed0ce88f109808f66eff23fdbddb9a87ef2bad. These are source deductions, not a reproduction of the original execution.

## What was actually saved

run5 exact/recipe108 JSON contains accuracy, Ops, elapsed wall seconds and four counters. Recipe also contains leafsum_minus_1. Neither contains checkpoint/weight bytes, dependency versions, command line, process CPU, per-cone errors, exact rays, RNG state or branch census. run5.py selects zoo(8,5,0); zoo44 converts the checkpoint's state tensors to float64 arrays. This is generator provenance, not proof the current installation matches the original runtime. MC JSON holds only .8357964; mc5.py documents10M Gaussian draws from default_rng(2026), chunk500k, float64 recurrence, first-index output argmax and second-ranked input argsort. Finite RNG input ties and platform behavior are not independently logged. No new checkpoint load or replay was performed.

The previous TASKS T185 claim requested five zoo seeds plus noisy copies. Only zoo0 is present. The README attributes stopping to a1e11-operation limit; that number is not present in the previous claim row inspected. The result is far above that limit, but its original briefing provenance is not independently recoverable from these files.

## Cost, timing and diagnostics

Ops is a hand-added abstract counter. m43.Ops describes multiply-add=1 and sqrt/division/atan2=1. leaderboard.fpe_flops instead scales an analytic scalar-FLOP expression to6000 atH16T10; H8T5 yields2500/3. This reproduces the historical comparison only. Neither complete actual native numerator nor a fully matched-unit denominator follows.

run5.py with any third argument wraps _cell_error and calls mass(c, Ops()) before returning the original _cell_error. That diagnostic mass is inside the timer but has a fresh discarded counter. orthant/_orth5 increments global STATS['simplices'] for BOTH diagnostic and estimator mass calls. Therefore recipe3641.6seconds and4,893,634simplices include diagnostics, while217,701,746,501Ops excludes their counter. Exact2622.2seconds has no saved leafsum field. The recipe appears slower in the saved timing despite fewer nominal Ops; this cannot rank pure estimator runtimes. Cell counters remain _cell_error invocations, not minimal true-output regions. The>=3433region count is not in these run JSON records; provenance requires another artifact before independent verification.

## Accuracy and certificate limitations

Both modes call the same _orth5 implementation and numerical geometry. Agreement is compatible with common bias. Leafsum close to1 is a conservation check, not an individual or signed-error bound: local errors can cancel while error-cell totals remain biased. The recipe still uses per-gate allowance checks and finite minority-ray/edge checks in _margin_split; the prior T108 simultaneous-error/downstream-kink qualifications therefore remain. This source review does not establish a visited-cell counterexample in T185.

_cell_error handles all-off outputs with first-index ownership, but otherwise clips every contrast using closed halfspaces with no explicit all-identical-coefficient winner ownership. Prior T60/T113 degenerate tie caveats remain; generic trained validity is a separate question.

## Five-dimensional validation

Independent T55 source review confirms, in exact nonsingular SPD arithmetic, the5D simplex prefactor3/(8pi^2), Plackett derivative/base constants, and conditional pair cofactors from the4x4 block given only(i,j). That does not enclose fixed-rule quadrature, input normalization, rounded cofactors/floors or final source sums. The prior wrong full5x5-precision implementation is preserved in the author's narrative, but its original failing raw rows are not among these14 snapshots.

val5.py computes each half z using one10M estimate in the numerator and a SECOND independent10M call for its standard error. The frozen value is a reasonable approximate diagnostic but is not a binomial z using its own observed hit count; neither component hit count is saved. Total documented validation MC is60M for six simplices plus60M for the three twice-sampled halves, rather than90M. Main-model MC adds10M. Failed earlier experiments are not covered by this source-derived total.

The saved half sum is +7.527312106958561e-14, whereas README/INDEX print a minus sign. All60 thin controls compare numerical rules; tight mutual agreement is not a rigorous error certificate. The test header's advertised determinant range need not match the actual saved range. Absolute and relative disagreement, negative low-order values, actual production-branch eligibility and all alternative rules are retained by audit.py.

## Scaling scope

Finite T3/T4/T5 observations do not imply recurring>=13.5growth throughT10, constant minimum ops/cell, or monotonicity in hidden width for independently trained weights. A constant continuation afterT5 matches all observed finite values, so a future lower bound needs algorithmic reasoning beyond that trend. This logical counterexample says nothing about the actual unknown T6..T10 runtime.

Prior T133 established separate completion bounds for the unchanged factorial chart recipe under its stated assumptions; those are legitimate narrower statements and not cancelled by this audit. Its reviewed chart_prefix_results.json reports2,090,188,800 first-prefix charges plus25,546,752,000 first-margin charges, total27,636,940,800 atH16T10. Those exclude deeper work/masses and assume completion of the unchanged chart/ray-testing architecture. They are distinct from the>=1e17 trend projection. Coarser charts, adaptive integration, shared work, alternative certificates and output-focused methods remain unexcluded.

## Preserved review corrections

Before any numerical execution, reading the saved T4 schema exposed that the recipe row is nested. Commitd4625b5d fixes that access. No failed numerical run or overwritten result was involved.
