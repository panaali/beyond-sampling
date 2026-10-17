# T156 finite native rank numeric envelopes

## Question
Can reviewed actual native-kernel call counts support a numeric source-work envelope without assuming a universal SVD iteration bound? This continuation owns only this new folder; previous T156/T165/T163 files remain unchanged.

## Method
Claim6023082d first pins the finite exact/recipe census, public LAPACK sources and installed binaries. Map parent/child calls before composition: DGESDD contains reductions and bidiagonal singular values; reductions call reflector/norm/BLAS helpers; DLASQ3 calls shift selection and observed DLASQ5. Price child work separately from parent-exclusive work to prevent double counting. The first bounded kernel is DLASQ5, whose actual active lengths are3/4 and whose scalar loops have explicit finite length. Use a source proof plus counted translation and frozen installed-library controls before contracting actual counts. Future helper/source envelopes require their own explicit evidence.

Alternative explanations: a finite census resolves iteration uncertainty but leaves per-call work unknown; public source and installed instructions can differ; kernel observers may include calls from outside rank. Never turn a partial named-category total into complete cost. In particular DNRM2 is a global symbol and can include geometry norms; audit attribution before composition.

## Status
DLASQ5 source proof independently checked,64 frozen installed controls pass bitwise, and actual histogram contraction complete. Full native envelope remains open; reflector-exclusive work is next.

## Results
DLASQ5 actual201773 exact/165964 recipe calls have conditional pinned-source numeric upper subtotals5949769/4879642. These count only Q5 arithmetic/negation/comparison, excluding index/control/memory and all other rank work. See Q5_DERIVATION.md, controls.json and q5_summary.json. Existing geometry/mass/model calculations are not rerun.

## Compute accounting
One thread2GB, review<=20CPU, primary<=110CPU. Cost convention counts scalar numeric arithmetic/comparisons/elementary calls explicitly; integer index/loop operations, memory/runtime and unpriced native helpers remain separately unknown. Ordinary source-operation envelope is neither a hardware count nor a complete rank cost until composition is finished.

## Claims
[inferred] Observed finite call lengths/counts can remove the need for a universal retry cap on these saved evaluations. [speculative] Conservative source envelopes will close enough numeric gaps to compare total costs.

## Alternatives not ruled out
Different native dispatch, uninstrumented helpers, replacement rank methods, looser declared source-operation upper bounds, and model-dependent setup. Existing T163 failure-cleanup reachability gap is not silently assumed away.

## Failure modes
Incorrect ABI, wrong census dimensions, parent/child double counting, omitted comparisons/negations, source/binary mismatch, treating control-flow statements as free numeric operations, assigning finite observed evidence to all inputs, or mistaking a subtotal for a total.
