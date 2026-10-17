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
rectangular diagonal and dyadic near-threshold matrices. Check both default
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
Protocol frozen; local source/dispatch inventory next.

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
