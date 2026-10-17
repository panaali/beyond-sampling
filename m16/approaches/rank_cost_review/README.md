# T156: values-only SVD/rank kernel cost audit

## Question
Can the remaining matrix-rank calls in T146/T153 receive a defensible
implementation-aware arithmetic envelope instead of an unpriced placeholder?

## Method
TASKS-only claim21e1d23 precedes implementation. Pin installed NumPy Python
matrix_rank/svd dispatch, linalg extension and linked OpenBLAS binary, plus
T146/T153 shape/call ledgers. Matrix shapes are real float64 Mx4, M in the
observed finite1..24 range; no trained matrix is re-evaluated.

Inspect actual values-only SVD entrypoints and public pinned source. Public
upstream source is evidence about that version; do not assume equivalence
to the installed0.3.34.106.0 build from the0.3.34 tag alone. Pin local symbols,
workspace-query and runtime options before deriving a count. Distinguish
finite arithmetic expressions, comparisons/control, copies/packing, BLAS
padding and compiler/hardware categories. If a relevant source path or loop
bound cannot be justified, preserve it as opaque.

Freeze dispatch controls on every observed Mx4 shape with zero, rank-one,
rectangular diagonal and dyadic near-threshold matrices, plus the same
diagonal scaled by2^500 and2^-500 to exercise both scaling directions. Check both default
matrix_rank tolerance and explicit1e-12, which are the source's two calls.
Compare baseline/interposed singular values/ranks bitwise and independently
known diagonal/rank-one answers where applicable. Record LAPACK job/dimensions,
workspace queries, numerical status and lower-level iteration paths without
changing arguments or results. No input sampling, geometry, mass replay,
new quadrature or replacement SVD is permitted.

Only after dispatch/source review derive a conservative finite arithmetic
bound for the observed shape set and scale by the existing exact/recipe
census. Iteration caps are bounds, not average measured iteration counts;
convergence failures remain visible. T150 independently owns4x4 inverse and
T144 determinant replacement, so neither is reimplemented here.

## Alternatives and falsifiers
1. Singular-values-only path is much cheaper than full SVD; job interception
   separates it from an unjustified generic cubic estimate.
2. Iterative bidiagonal work has finite enforceable caps; source/binary loop
   limits must support them rather than relying on toy convergence speed.
3. Runtime dispatch/packing differs from public reference source; retain
   platform ambiguity and do not promote a hypothetical algorithm's cost.
4. Conservative bounds are too loose for the candidate budget; retain this
   as a bound limitation, not a proof that the implemented kernel is costly.

## Status
Dispatch/control phase complete; internal iteration and numerical-cost envelope remain open.

## Results
Pending. No complete portable/hardware cost or estimator win is claimed.

## Compute accounting
One thread,2GB, <=20CPU review or <=110CPU primary bounded phases. Include
source inspection, compilation, control and failed work separately. No
neural-model evaluations or Gaussian integrals.

## Claims
[inferred] Source values-only rank calls need their actual branch examined.
[speculative] A conservative small-dimension bound can close a major numerical
cost category while leaving interpreter/control/memory qualifications explicit.

## Alternatives not ruled out
A counted rank replacement, exact incidence-based rank proofs, other numerical
libraries/platforms, and reducing geometry calls. These require separate scope.

## Failure modes
Assuming exact upstream equivalence, counting full SVD when vectors are absent,
ignoring workspace/scale/iteration failure paths, incomplete interception,
mispricing padded kernels, and treating arithmetic as hardware instructions.

Initial inventory pins NumPy2.5.4 and the same OpenBLAS binary as T150.
Installed ILAVER reports LAPACK3.12.0. matrix_rank calls non-Hermitian
svd(compute_uv=False); the double wrapper imports scipy_dgesdd_64_.
Disassembly records separate workspace-query and execution call sites,
with the NumPy14-argument ABI. Fixed interposition forwards those exact
arguments, unchanged, to the installed binary. No envelope has yet been
inferred from this dispatch observation. Public OpenBLAS0.3.34 source is
pinned separately and exact installed-source equivalence remains unproved.

The 114 frozen analytic cases (19 observed shapes times six matrices) have
known ranks under both tolerances. Baseline and pass-through results agree
bitwise for all singular values and exactly for both ranks. Each case calls
SVD three times, each with one workspace query and one execution; all 684
intercepted calls use JOBZ=N and return INFO=0. Large/small scaled diagonal
controls exercise scaling without trained input replay. Baseline and spy CPU
were 0.029987 and 0.034970 seconds. These are finite dispatch controls, not
a bound on iteration counts for trained matrices. Compile the review-only
spy with `cc -shared -fPIC -O2 spy.c -ldl -o spy.so`; the binary is not committed.

Source inspection finds DBDSDC values-only dispatches through DLASDQ.
The actual nested bidiagonal path and retry bounds must be checked before
composing a cost envelope. Public source alone does not prove installed
implementation identity. No full-cost gate is unblocked by this phase.

### Frozen nested-path continuation
Reuse the same 114 matrices only to distinguish the newly identified
DLASQ1/DLASQ2 nested path; this is a different dispatch discriminator,
not another estimate of convergence speed. A separate pass-through spy
records calls/orders/status and leaves all operands/results untouched.
Require exact agreement with saved baseline values/ranks. For min(M,4)<=2,
DLASQ2 must never be entered; its occurrence falsifies the source branch
interpretation. For higher orders some rank-one fixtures should enter it,
but no fixed iteration upper bound is inferred from that observation.

Nested-path pass-through retained all 114 baseline results. However, the
first diagnostic attempted to read DLASQ2's iteration-output workspace on
every successful return; early returns do not populate that slot, producing
invalid integer counters. Preserve `nested_instrumented.json` as the failed
diagnostic artifact. This invalidates only its iteration fields, not numerical
parity. The repair counts actual DLASQ3 entries and its documented integer
ITER increment across each call instead of reading unspecified workspace.

The repaired counter observes 324 DLASQ1 calls, 48 DLASQ2 calls, and 192
DLASQ3 calls on the same panel, with zero dqds iterations: rank-one controls
deflate without the dqds update. All numerical outputs still match exactly.
This does not validate the iterative branch. Before running that branch,
freeze three new positive tridiagonal controls at M=3,4,24, with square
leading K=min(M,4) block diag2/offdiag1 and remaining entries zero. Their
singular values are 2+2*cos(j*pi/(K+1)); both rank tolerances must return K.
Require baseline/pass-through bitwise equality and positive recorded dqds
iterations on at least one fixture, otherwise iterative path remains untested.

Three tridiagonal controls pass analytic singular values (max discrepancy
1.34e-15), known ranks and exact baseline/interposed parity. They record
78 actual DLASQ3 calls and 69 dqds iterations across nine SVD executions,
max one iteration per call. This validates the counter on nontrivial finite
inputs; no worst-case bound follows. CPU baseline0.024395/interposed0.027833.

Reproduction (one thread and review lock as in task protocol):
`cc -shared -fPIC -O2 counted_spy.c -ldl -o counted_spy.so`; then
`python dense_controls.py baseline` and
`LD_PRELOAD=$PWD/counted_spy.so python dense_controls.py instrumented`.
`counted_controls.py instrumented` under the same preload reproduces the
114 original controls with the repaired counters.

### Independent review and bounded handoff
T55 c4fed84f independently checks all114 analytic cases, the3 dense spectra,
684 GESDD calls and78 Q3/69 actual iterations. T163 d7dd46ac independently
finds a conditional four-Q5-attempt proof but keeps active-array invariants
and installed-source equivalence explicit. Public Q2 failure cleanup has a
non-progress syntactic state; reachability from valid inputs and installed
binary behavior are unproved. Consequently no universal finite SVD cost
envelope is claimed here. This is an audit limitation, not an estimator or
SVD-method negative. The distinct claimed rank_runtime_census folder (T165
claim cb6ed1c0, duplicate numerical ID pending DRI reconciliation) records
actual finite evaluation paths instead. Native reduction bounds and all
unpriced numeric helpers remain part of the overall cost gate.
