# T156 finite native rank numeric envelopes

## Question
Can reviewed actual native-kernel call counts support a numeric source-work envelope without assuming a universal SVD iteration bound? This continuation owns only this new folder; previous T156/T165/T163 files remain unchanged.

## Method
Claim6023082d first pins the finite exact/recipe census, public LAPACK sources and installed binaries. Map parent/child calls before composition: DGESDD contains reductions and bidiagonal singular values; reductions call reflector/norm/BLAS helpers; DLASQ3 calls shift selection and observed DLASQ5. Price child work separately from parent-exclusive work to prevent double counting. The first bounded kernel is DLASQ5, whose actual active lengths are3/4 and whose scalar loops have explicit finite length. Use a source proof plus counted translation and frozen installed-library controls before contracting actual counts. Future helper/source envelopes require their own explicit evidence.

Alternative explanations: a finite census resolves iteration uncertainty but leaves per-call work unknown; public source and installed instructions can differ; kernel observers may include calls from outside rank. Never turn a partial named-category total into complete cost. DNRM2 is a global symbol without parent attribution; geometry norm overlap is possible, not observed proof. Audit attribution before composition.

## Status
DLASQ5 source proof independently checked,64 frozen installed controls pass bitwise, and actual histogram contraction complete. Reflector-exclusive113-unit generic source bound and160 native-child wrapper controls also pass. Resumed under claim8c9d87a4 after the T179 handoff. Binding controls and saved phase equality support a conditional finite reflector-exclusive10 bound (generic113 retained). Installed DNRM2 source gate and192 controls pass; cold-once raw contraction independently reviewed d90477c1. Full native envelope remains open.

## Results
DLASQ5 actual201773 exact/165964 recipe calls have conditional pinned-source numeric upper subtotals5949769/4879642. These count only Q5 arithmetic/negation/comparison, excluding index/control/memory and all other rank work. See Q5_DERIVATION.md, controls.json and q5_summary.json. Existing geometry/mass/model calculations are not rerun.

DNRM2 installed named numeric subtotals are6,495,956 exact and5,889,672 recipe, including28,696 cold units once. Its wrapper uses one abs for N=1; five bundled kernels use2N+4 units for N>1 positive stride. These are x87 extended scalar units, not hardware cycles. All192 controls pass (.008 CPU); norm_review/contraction.json preserves all raw row hashes. Binding provenance remains an explicit condition for the separate finite no-rescale inference.

Helper controls64 DLAPY2 +20 machine constants +320 DSCAL all pass (.020 CPU). Installed source subsets13 perhypot and N+3 perscale give a conditional reflector/norm/helper upper31,689,663 exact /29,160,578 recipe. Generic up-to20-rescale counterpart230,514,579 /212,199,418 remains explicit. Saved helper source/control/contraction independently reviewed0cfdfa39; norm source/raw independentlyd90477c1. None includes other SVD work or unpriced runtime/control categories.

Shift selection Q4 now independently reviewed3ed60f55:1152 bitwise controls cover all12 source shift types, named-source upper9,358,549 exact /7,673,352 recipe from saved Q3 entry lengths. Q3 parent864 controls passed but initially exercised no retries; separate16 frozen controls then passed with up to3 updates, initial coverage limitation preserved. Q3 parent-source upper18,381,417 /15,126,366 is saved and independently checked in f4b76ba3. It uses actual Q5 counts, not a universal retry claim. Retry CPU reported0.0 at process-timer resolution, not an assertion of zero work.

## Compute accounting
One thread2GB, review<=20CPU, primary<=110CPU. Cost convention counts scalar numeric arithmetic/comparisons/elementary calls explicitly; integer index/loop operations, memory/runtime and unpriced native helpers remain separately unknown. Ordinary source-operation envelope is neither a hardware count nor a complete rank cost until composition is finished.

## Claims
[inferred] Observed finite call lengths/counts can remove the need for a universal retry cap on these saved evaluations. [speculative] Conservative source envelopes will close enough numeric gaps to compare total costs.

## Alternatives not ruled out
Different native dispatch, uninstrumented helpers, replacement rank methods, looser declared source-operation upper bounds, and model-dependent setup. Existing T163 failure-cleanup reachability gap is not silently assumed away.

## Failure modes
Incorrect ABI, wrong census dimensions, parent/child double counting, omitted comparisons/negations, source/binary mismatch, treating control-flow statements as free numeric operations, assigning finite observed evidence to all inputs, or mistaking a subtotal for a total.
