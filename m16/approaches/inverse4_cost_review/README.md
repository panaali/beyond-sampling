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

## Results (bounded phase completed)

The installed path and all11 fixed fixture outputs/exceptions are independently checked. GESV receives n=nrhs=lda=ldb=4 with identity RHS. Seven nonsingular controls also pass exact-rational inverse checks. Process-local kernel interception reproduces baseline values bitwise and restores all pointers. Pivot, singular, near-singular, subnormal and batch paths are preserved. No trained matrix or mass calculation was rerun.

`DERIVATION.md` derives a188 floating-expression envelope:56 LU plus132 packed triangular solve/packing operations. `summarize.py` contracts saved kernel events and composes the T146/T153 inverse call censuses, pinned by file hash. Exact/recipe inverse subsets are500,062,328 /402,044,768 units. This is not a complete inverse or implementation cost, and it does not establish an MSE win. Source provenance records public upstream version versus installed binary qualification.

Reproduce saved-data recount: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/clean-neurons/.venv/bin/python m16/approaches/inverse4_cost_review/summarize.py` (0.0024CPU). The original fixture runs used `fixtures.py baseline`, then `LD_PRELOAD=<folder>/spy.so fixtures.py instrumented`; `kernel_controls.py` performs the separate table wrapper check. Compile shared wrappers using `gcc -shared -fPIC -O2 <source>.c -ldl -o <target>.so`. Existing numerical results are preserved; no repeat is required for the count.

Measured fixture CPU: baseline0.024936, GESV passthrough0.023848, kernel passthrough0.041183. Import/dispatch/source inspection is additional setup; these are not estimator timing claims. T55 independently checked all11 saved outputs, dimensions/events and count algebra without rerunning inverses; review commit pending.

## Alternatives not ruled out

A complete control/category upper bound, a separately evaluated replacement inverse, or another backend may change the total. Larger matrices and thread counts do not inherit188. Memory/allocation, comparison/abs, integer/control/index and wrapper work remain unpriced, as do other parent opaque categories. This bounded result is positive for identifying the FP subset, and incomplete for closing the common-budget gate.

## Final status and claim scope

Done for this bounded floating-expression audit; independent review passed (commit to follow). [measured] Installed dispatch and transparent fixture parity. [inferred, independently checked]188 fixed-path floating-expression envelope and composed category totals. [speculative] A full category envelope may still be comfortably below sampling break-even; no complete-cost or challenge-solved claim is made. Initial progress notes above preserve the chronology and are superseded by this final status.

Independent T55 review **79b6876** verifies all11 saved passthrough controls, event-based188 derivation and scope exclusions. Parent T153 raw aggregate also passes that review.
