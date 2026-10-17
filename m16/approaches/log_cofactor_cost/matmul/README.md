# Installed array-multiply dispatch continuation

## Question
Which native matrix paths actually execute in the pinned mass source, and can their numeric interiors replace the earlier scalar-expression charges without silently changing the implementation? RenewedTASKS-only5b25503f precedes implementation. Parent bounded phaseba14964d is independently26fd4e11; preserve its original hybrid subtotal.

## Method
Public NumPy2.5.4 source explicitly chooses SYRK for identical-pointer selftranspose products and GEMM otherwise. Installed binary contains both dispatch calls at the listed addresses, so generic GEMM-only pricing would be unjustified. Source/binary hashes and provenance are in manifest.json; no full upstream build-equivalence claim. Exact source products are B@B.T, N@N.T (4x4) and DU@Rays (1000x4 by4x4). Do not assume shape alone specifies stride/layout or symmetry dispatch.

Freeze12 synthetic integer-array fixtures: 4x4 identity selftranspose; dense C-order selftranspose; dense F-order selftranspose; strided4x4 selftranspose from an8x8 parent; independent denseC product; mixedF/C independent product; and1000x4 zero, ones, dense modular-integer, transposed right operand, F-order left operand, and strided left operand. Dense square uses[[4,1,2,0],[1,5,0,1],[2,0,6,1],[0,1,1,3]], tall dense uses arange modulo7 minus3. Keep sharing/layout in each fixture. Independent reference is exact integer multiplication. Transparent CBLAS GEMM/SYRK wrappers must preserve output bits and return exact references, record dimensions/transposes/leading dimensions/alpha/beta, and make no actual source mass calls. Source review before controls. Synthetic values test dispatch/parity only and do not prove all model-dependent branches or complete kernel cost.

After dispatch gate, identify/preregister relevant installed small kernels before any operation-price claim. Do not erase triangle-copy memory or generic setup: state whether excluded by the numerical convention. Source matrix expressions and actual packed lanes are different objects. Strided inputs may require copies and lose the shared-pointer optimization; preserve that outcome rather than assume all products use SYRK.

## Status
All12 dispatch/parity controls complete; independent raw review155b3b18 passes.

## Results
All12 baseline/interposed products agree bitwise and with exact integer matrix products. Contiguous C/F selftranspose fixtures (three cases) dispatch SYRK. The strided selftranspose case is copied and dispatches GEMM with distinct pointers; all other independent/tall cases dispatch GEMM (nine total). Alpha1/beta0 and every dimension/transpose/leading dimension are saved. The six tall cases retain1000x4 by4x4 geometry, including leading dimension8 for the strided tall left matrix. No operation price has changed; actual child-kernel accounting remains open.

Reproduce baseline and LD_PRELOAD=<absolute spy.so> instrumented controls under one-thread2GB review lock. CPU .093037018+.079143817=.172180835, including exact checking but excluding imports; no mass/model runs.

## Compute accounting
One thread2GB, review lock<=20CPU each. No cone/model/MC/reference integration. Fixed12 analytic array products only; exact integer checking and instrumentation validation costs separately recorded. No timing-to-operation conversion.

## Claims
[inferred] Source symmetry predicate can select a cheaper triangular route; actual paths/costs remain pending. No fullcost/MSE or accuracy implication.

## Alternatives not ruled out
Copying/layout can disable symmetry (inspect actual pointers/strides); installed build may differ from public source (binary plus transparent dispatch); GEMM/SYRK may have unexpected packed tails (capture actual child kernels before pricing). Wider shapes/threading/backends untested.

## Failure modes
Replacing installed code with mathematical minimal dot counts; reusing GESV child counts for different GEMM/SYRK paths; unsupported signatures; changed output due to instrumentation; incomplete native/control/memory accounting; extrapolating12 synthetic cases to all possible arrays.

## Frozen child-kernel capture
Independent155b3b18 validates all12 initial dispatch fixtures. Runtime read-only inventory reveals direct small-GEMM kernels, so the older GESV-child wrapper alone would omit a relevant path. Freeze one new capture of the same12 layouts for child identity/shapes; this is additional instrumentation evidence, not a repeated mass/model experiment. kernel_spy.c wraps exactly16 pinned table entries, records bounded4096 events, returns original results and restores pointers in finally. Pinned runtime targets checked before mutation. Small normal/beta0 signatures are matched across all four public NN/NT/TN/TT sources; installed CBLAS disassembly shows beta0 selection. Static SYRK helpers call ordinary GEMM/BETA children and remain a separate source-overhead obligation.

No numeric kernel price is asserted before actual path capture and binary reasoning. In particular, source NN M4,K4 tail has masked loads but unmasked8-lane arithmetic, potentially charging unused lanes; source alone does not establish the selected path or installed instruction count. Keep model-free dimension-dispatch arithmetic and model-dependent arithmetic categories separate. No revised corpus subtotal yet.

Read-only dispatch inventory initially used .058700918 CPU. A second read adds the previously unrecorded runtime GEMM_UNROLL_MN field at0x304, needed to inspect SYRK diagonal blocks; both costs are retained rather than replacing the initial inventory record.

The requested refuse-existing guard edit initially used a repo-relative path from the subfolder and raised FileNotFoundError; the shell continued to the first capture, which succeeded. Its result/log were new (noclobber log protection), so no earlier evidence was overwritten. The file guard is now installed and no repeat is performed. Preserve this orchestration error separately from numerical validation.
