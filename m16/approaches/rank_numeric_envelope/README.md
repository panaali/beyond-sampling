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

Outer Q1/Q2 bounded phase now passes630 source/native helper controls (216 DLAS2,48 positive-finite DLASCL,364 small sorts,2 REAL32 IEEECK), .036012CPU. All bitwise parity and named bounds pass, including scaling extremes. The static cold eigensolver exclusion received independent source review: leggauss uses DSYEVD JOBZ=N→DSTERF, while remaining rank matrices have four columns. The original lastN gap remains preserved; this is a separate pinned ABI/callgraph argument, with historical binding premises explicit. Outer saved-census upper25,958,983 exact /23,059,486 recipe includes cold once and excludes Q3/Q4/Q5. Raw control/contraction independently passed fd0dabc7. Reproduce with outer_review/helper_controls.py and outer_review/contract.py only in a fresh result-free checkout; commands refuse overwrite.

The disjoint priced subset is now91,338,381 exact /79,899,424 recipe using conditional no-rescale reflector pricing; generic reflector alternative290,163,297 /262,938,264. These remain incomplete named numeric source/native subsets: matrix reductions and their BLAS, remaining wrappers/helpers, integer/control/memory/runtime are not fully priced. They are not complete cost bounds or sampling/MSE conversion factors.

Reduction application phase: installed DGEQR2/DGELQ2/DGEBD2 call unprefixed dlarf1f_; symbol correction and pre-run right-scan bound correction preserved. Independently source-gated1152 wrapper controls all pass bitwise matrix/work/V and padding checks, .107968CPU. Controls share original BLAS children, so validate parent semantics only. Side-specific scan/negation upper contracts855,533/795,427 application calls to10,520,049 exact /9,590,329 recipe named source units, cold once. No DLARFG/BLAS child double-counting; raw independent review passed365823b1. Reproduce reduction_review/controls.py and contract.py in a fresh result-free checkout,20CPU/2GB/one thread. Partial priced subset becomes101,858,430 exact /89,489,753 recipe (generic reflector300,683,346/272,528,593). BLAS matrix/vector bodies and remaining drivers/control/runtime remain open; no full-cost bound.
