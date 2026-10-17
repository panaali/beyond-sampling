# Latest log/cofactor mass source-cost audit

## Question
What numerical work does the actual139a7b9e four-dimensional mass implementation perform, including its thin1000-node ray-space branch and wide128-node cofactor/atan2 branch? TASKS-only claim da953e87 was pushed first. Historical malformed task ID T11579 is retained pending DRI reconciliation; identify this task by folder/title.

## Method
Pin unchanged source and previously reviewed installed NumPy/OpenBLAS dependencies. First validate the actual3x3 determinant dispatch, numerical categories and source-visible branch arithmetic. Reuse reviewed4x4 determinant176 and inverse290 named numeric-category bounds only after matching binary hashes and actual dimensions. Keep category upper bounds separate from counts and full-cost claims. Full35098 census contraction waits independent T179 raw review; no mass, geometry, model or truth replay.

Freeze the following18 three-dimensional determinant controls before execution: identity; positive diagonal(1,2,4); negative diagonal(-1,-2,-4); one-swap identity; dense symmetric[[4,1,2],[1,5,0],[2,0,6]]; three-cycle rows; near-singular upper2x2 block with epsilon2^-40; zero; duplicate row; subnormal diagonal2^-1023; dense scaled2^-250/2^250; overflow/underflow diagonals2^400/2^-400; balanced diagonal2^500,2^-500,1; NaN diagonal; infinity diagonal; two-matrix batch. Compute exact finite rational determinants independently of floating LU, preserve nonfinite/singular/overflow/underflow cases. Transparent GETRF/log/exp interposition must preserve baseline returned float bits and exceptions. Source/dispatch shape review precedes any aggregate assertion.

Tentative3x3 category derivation: LU arithmetic24, absolute/comparison lanes63, reconstruction17, total104. This is a hypothesis until actual small-path shapes and source guards are checked. Small LU threshold alone is insufficient evidence of all downstream shapes. No substitution of a direct3x3 determinant or removal of the executed initial4x4 inverse is allowed.

For visible source costs, state a consistent arithmetic-expression convention; inventory each scalar/vector/matrix dot, reduction, norm, transcendental and comparison. Record model-free reusable rule generation separately. Do not add the inherited mass counter to newly recounted operations (double counting); retain old totals for comparison. Numeric-library internals, integer/control, memory/allocation and setup exclusions remain explicit. No extrapolated multiplier from old mass rules or other estimator panels.

## Status
Bounded source/control/contraction complete; independent final raw review pending. T179 latest full census c05a4bd2 independently reviewed4cc3fe39.

## Results
All18 frozen fixtures (19 matrices) preserve determinant outputs bitwise under GETRF/log/exp interposition and process-local child-kernel capture. Finite exact-rational comparisons pass, including retained singular/subnormal/overflow/underflow/nonfinite cases. The observed19 GETRF calls have dimensions/lda3,51 logs and19 exps;17 successful and2 singular matrices. Actual runtime pointers/block parameters match the pinned prior dispatch. Child shapes match the proposed3x3 bounds with no GEMM/solve path. This supports the104-unit named numeric subset, not a full library cost.

The reviewed35,098-cone census contains17,955 thin and17,143 wide calls. Source-visible arithmetic978,197,934 plus small-native determinant/inverse category envelopes1,800,324,820 gives hybrid2,778,522,754. The inherited mass counter is1,064,714,168; their2.60964 ratio is only an accounting diagnostic, not a cost correction multiplier for MSE. Model-free runtime constants190,197 remain separate. See VISIBLE_LEDGER.md, DETERMINANT3.md and summary.json for categories. These are exact-source mass calls, not a full recipe or estimator count. No mass/model/MC reevaluation occurred.

Reproduce: compile spy.c with gcc -shared -fPIC -O2 -ldl and kernel_spy.c likewise; run controls.py baseline, LD_PRELOAD=<absolute spy.so> controls.py instrumented, and controls.py kernels under the review lock with one thread2GB. summarize.py contracts saved controls/census. Control CPU .024640942+.028443582+.040779073=.093863597; contraction .002191372, imports/compilation separate. No unplanned repeated numerical runs.

## Compute accounting
One thread2GB; bounded analytic controls under20CPU review lock, source/count work only. No mass/model/geometry/MC replay. Instrumentation is validation work and not inference overhead. Elementary function calls count as named operations while libm internals remain unpriced; category envelopes are not hardware instruction counts.

## Claims
[inferred] Previously reviewed4x4 native categories may transfer after exact binary/dispatch checks. [measured/inferred] Frozen actual-dispatch controls and source/lane analysis support3x3 numeric subset104, pending independent final raw review. No full-cost superiority, quadrature accuracy or rung claim.

## Alternatives not ruled out
Different small-size dispatch or packed lanes could invalidate naive dimension scaling: inspect binary and actual shapes. Source/counter mismatch could explain inherited low costs: reconstruct branch arithmetic directly. Thin and wide costs differ: retain both multiplicities and no mean multiplier. Reusable setup might be free under protocol but model-dependent executed inverse is not: separate rather than erase it.

## Failure modes
Double counting inherited counters; counting only mathematical nonzeros in dense operations; silent source/backend drift; singular and subnormal paths; integer tuple/branch comparisons mixed with a numeric-only sampler; conflating a bounded category upper with a full implementation upper/lower. Preserve any failed parity/control and unresolved categories. This is no method-family negative.

## Explicit remaining native interiors
Visible dense dot/reduction/norm charges are scalar-expression counts; installed determinant/inverse categories are numerical-lane envelopes. BLAS accumulator work, NumPy reductions and Python3.12 sum compensation are not automatically covered by the visible convention. Report their combination as a hybrid subtotal with these gaps, never a full native upper/lower or a matched-budget MSE result.
