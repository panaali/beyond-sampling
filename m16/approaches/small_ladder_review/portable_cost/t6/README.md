# T52 counted independent H2T6 DD geometry

## Question
Can independent DD geometry reduce the H2T6 simplex count enough to support a fully counted deductive integration comparison?

## Method
Parent pushed extension b7209a1 before this work. Edit only portable_cost/t6; preserve T4/T5. Begin with base seed0 geometry only. Read the checkpoint directly and derive gates, first-index output regions, second-argmax regions, orthant cuts and pulling triangulation. Do not use author ray caches or select regions from author results. Reuse the audited DD wrappers with width6 bounded rank and scalar6x6 determinants.

Constraint IDs are explicitly disjoint: gates0..66 for H2T6, output80..85, target-rank86..91, orthants100..105, all within two64-bit words. T5's hard-coded output IDs are unsuitable here. Record every ambiguity/failure with partial work and exact failing matrix. No opaque numerical kernel in production; any SVD diagnostic is separate.

## Status
Frozen protocol before implementation. First produce complete gate and correct-region geometry with full scalar ledger, CPU, simplex counts and deterministic affine checks. The orthant L1 cross-section determinant sum of the full gate partition must equal64 within declared floating diagnostic tolerance. This is an empirical partition check, not an exact geometry certificate. Only after reporting these results will triangle integration be planned.

## Results
Pending. Frozen external comparison only: author Qhull route has136438 correct simplices and accuracy0.24815022292589073 with conditional bound9.04235e-6. Do not use that accuracy to guide geometric decisions or drop regions.

## Compute accounting
Charge all gate/output/rank/orthant clipping, adjacency tests, face traversal, width6 bounded rank, normalization and6x6 determinants. Report gate-partition validation work separately from production correct-region work. Scalar arithmetic/Boolean/64-bit bitset operations cost1; transcendental cost20. Memory copies/addressing, containers and interpreter overhead remain excluded from the arithmetic metric but included in CPU. Save geometry and source hashes. One thread,2GB memory, <=110CPU per cycle, common compute lock for potentially long geometry. No random model inputs and no numerical mass integral in this phase.

## Claims
[speculative] DD may substantially reduce author Qhull triangulation size. [inferred] This geometry pipeline preserves the tested T4/T5 structure but needs dimension6 validation. No efficiency or solved claim yet.

## Alternatives not ruled out
1. A smaller count might omit or duplicate regions through constraint collisions or rank decisions. Falsifier: explicit ID-disjointness assertions, full gate determinant sum64, retained rank ambiguity and shape/count diagnostics.
2. Width5 counters/routines might silently undercount or mis-handle width6. Falsifier: scalar determinant and rank primitives tested on fixed6D fixtures, complete operation components saved, and no unsupported width accepted.
3. Floating tolerances may corrupt narrow cones. Falsifier: save any ambiguous matrix, compare separately with SVD, retain failures; partition and deterministic gate-map checks provide additional evidence but not a full roundoff certificate.

## Failure modes
Floating DD tolerances and rank brackets, combinatorial growth, finite bitset IDs, long exact determinant sums, and eventual triangle-integration setup/roundoff cost. Formal geometry certification remains a separate T68 workstream. No author cache reuse, region omissions or silent rank fallbacks.
