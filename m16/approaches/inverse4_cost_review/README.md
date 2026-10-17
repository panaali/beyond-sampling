# T150 independent fixed-4x4 inverse cost audit

## Question
What work does the actual installed NumPy inverse perform for the fixed4x4 calls in T146's repaired T4 cone calculation? Shared TASKS-only claim7ff2de4 precedes implementation. T146 supplies its parent source/census; T144 independently owns determinant accounting. No mass replays or source replacement.

## Method / frozen bounded phase
Pin installed Python/NumPy/OpenBLAS shared objects, Python dispatcher and local binary dispatch. T146 census eee3da5 has35098 covariance inverses plus2624760 transformed-matrix inverses:2659858 total; geometry48 expected calls remain pending. Pin estT SHAafa94627ff5b860a4b922dc340e90d0d6718c770d74e326dfbf667a2bb0bbc3a and T140 runner SHA38213c96b31538b0aad2532a9c169fad60db3725d8345a39f7782d33e6531d18.

Initial read-only inventory: NumPy2.5.4 double inv dispatches to _umath_linalg.inv; installed binary calls scipy_dgesv_64_. Linked OpenBLAS0.3.34.106.0 USE64BITINT uses dgetrf_single then dgetrs_N_single for small matrices. These observations do not by themselves justify a scalar cost for implementation-specific kernels. Capture hashes/disassembly and bounded independent call interception on fixed analytic fixtures to verify dimensions/right-hand sides/pivot/singular handling. Derive mathematical arithmetic counts separately from actual dispatch/packing/control work. Any unbounded category remains explicitly unresolved.

Alternatives: standard146-arithmetic Gaussian elimination is only a possible algorithm, not necessarily this OpenBLAS kernel (check actual small-size dispatch and padding); a numerical timing is not an operation bound (do not substitute it); singular/pivot/zero-skipping behavior may vary (fixed identity/diagonal/permutation/dense/near-singular/singular controls). No data-dependent search or sweeping trained matrices.

## Status
Source/dispatch inventory in progress; no inverse operation cost asserted. Parent geometry census and independent review pending.

## Compute accounting
One thread,2GB, <=20CPU review or <=110CPU primary phases. Capture import/tool/setup time separately. Explicit scalar arithmetic, comparisons, packed/vector padding, copying and platform details must be distinguished. Do not promote a replacement algorithm's count to the installed implementation; no full hardware or sampling claim. Any downloaded public upstream source is pinned and compared with installed binary provenance before relying on it.

## Claims / failure modes
[measured] Local binary links a pivoted linear-solve entrypoint. [speculative] A small fixed-dimension arithmetic envelope will provide useful headroom for the common budget. Dynamic-architecture kernels, compiler transformations and absent exact source may prevent a defensible complete bound; record this limitation rather than silently assigning a textbook formula. Determinant, SVD/rank and geometry call counts remain separate owners/scopes.
