# Installed array-multiply dispatch continuation

## Question
Which native matrix paths actually execute in the pinned mass source, and can their numeric interiors replace the earlier scalar-expression charges without silently changing the implementation? RenewedTASKS-only5b25503f precedes implementation. Parent bounded phaseba14964d is independently26fd4e11; preserve its original hybrid subtotal.

## Method
Public NumPy2.5.4 source explicitly chooses SYRK for identical-pointer selftranspose products and GEMM otherwise. Installed binary contains both dispatch calls at the listed addresses, so generic GEMM-only pricing would be unjustified. Source/binary hashes and provenance are in manifest.json; no full upstream build-equivalence claim. Exact source products are B@B.T, N@N.T (4x4) and DU@Rays (1000x4 by4x4). Do not assume shape alone specifies stride/layout or symmetry dispatch.

Freeze12 synthetic integer-array fixtures: 4x4 identity selftranspose; dense C-order selftranspose; dense F-order selftranspose; strided4x4 selftranspose from an8x8 parent; independent denseC product; mixedF/C independent product; and1000x4 zero, ones, dense modular-integer, transposed right operand, F-order left operand, and strided left operand. Dense square uses[[4,1,2,0],[1,5,0,1],[2,0,6,1],[0,1,1,3]], tall dense uses arange modulo7 minus3. Keep sharing/layout in each fixture. Independent reference is exact integer multiplication. Transparent CBLAS GEMM/SYRK wrappers must preserve output bits and return exact references, record dimensions/transposes/leading dimensions/alpha/beta, and make no actual source mass calls. Source review before controls. Synthetic values test dispatch/parity only and do not prove all model-dependent branches or complete kernel cost.

After dispatch gate, identify/preregister relevant installed small kernels before any operation-price claim. Do not erase triangle-copy memory or generic setup: state whether excluded by the numerical convention. Source matrix expressions and actual packed lanes are different objects. Strided inputs may require copies and lose the shared-pointer optimization; preserve that outcome rather than assume all products use SYRK.

## Status
Source inventory complete;12 controls frozen, no matrix test runs yet.

## Results
None. Both source and binary have SYRK and GEMM branches; actual captured fixtures pending.

## Compute accounting
One thread2GB, review lock<=20CPU each. No cone/model/MC/reference integration. Fixed12 analytic array products only; exact integer checking and instrumentation validation costs separately recorded. No timing-to-operation conversion.

## Claims
[inferred] Source symmetry predicate can select a cheaper triangular route; actual paths/costs remain pending. No fullcost/MSE or accuracy implication.

## Alternatives not ruled out
Copying/layout can disable symmetry (inspect actual pointers/strides); installed build may differ from public source (binary plus transparent dispatch); GEMM/SYRK may have unexpected packed tails (capture actual child kernels before pricing). Wider shapes/threading/backends untested.

## Failure modes
Replacing installed code with mathematical minimal dot counts; reusing GESV child counts for different GEMM/SYRK paths; unsupported signatures; changed output due to instrumentation; incomplete native/control/memory accounting; extrapolating12 synthetic cases to all possible arrays.
