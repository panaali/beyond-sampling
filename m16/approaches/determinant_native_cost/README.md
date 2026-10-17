# T158 actual installed determinant cost

## Question
What numeric work does the actual NumPy float64 four-dimensional determinant execute in T146/T153? TASKS-only claim7f18b7d was pushed before implementation. This audits the existing source; T144's alternative determinant is not substituted. Inverse and rank kernels remain separately owned.

## Method
Pin installed NumPy2.5.4 extension and OpenBLAS shared objects against T150 hashes. Read public NumPyv2.5.4 umath_linalg.cpp, record URL/date/hash, compare relevant compiled instructions. Installed det<double,double> calls GETRF then four log calls, log accumulation and one exp/sign multiplication; it is not a four-diagonal product. Independently intercept GETRF and log/exp transparently for fixed controls, preserving baseline outputs and exceptions. Verify GETRF small-size dispatcher shares the reviewed T150 small LU path before reusing its56 floating-expression envelope. Keep T157 pivot/abs/comparison categories separate and unknown until reviewed.

Freeze controls before numerical execution: identity; positive/negative diagonal; one-swap permutation; fixed symmetric dense; three-cycle rows; fixed near-singular; zero; duplicate row; subnormal diagonal; dense scaled2^-250 and2^250; overflowing/underflowing determinant diagonal2^300/2^-300; balanced diagonal exponents500,-500,400,-400; NaN diagonal; infinity diagonal; two-item batch. Exact rational determinants for finite controls and relative1e-12 normal-range comparison; preserve expected overflow/underflow/singular cases. All instrumentation must return unchanged outputs. No trained matrices, mass calculations or parameter changes.

Compose only existing35098 exact/28000 recipe determinant counts, pin source summaries. Report floating arithmetic, numeric comparisons/negations, transcendental calls and integer pivot work separately. libm internal implementations, Python/ufunc/control/memory/index/allocation costs remain explicitly unpriced. A named elementary-function count is not a hardware-FLOP bound.

Alternatives: actual determinant may use a different LU dispatcher than inverse (inspect installed binary); source version may not match installed code (binary comparison plus transparent controls); nonfinite/singular paths can change counts (fixed adversarial controls). No timing substitutes for count evidence.

## Status
Protocol/source inventory frozen; controls next.

## Results
No numerical controls yet.

## Compute accounting
One thread2GB, review lock<=20CPU each bounded process. No geometry/mass replay. Audit wrapper instrumentation is review-only, not charged as estimator inference. Numeric expression categories and library work stay separate; no complete portable-cost or sampling claim.

## Claims
[inferred] The determinant wrapper adds logarithmic reconstruction work beyond LU. [speculative] A checked fixed-dimension envelope can replace the opaque determinant category in the current partial cost accounting.

## Alternatives not ruled out
Other backends, replacement kernels and complete library/control accounting. No performance or estimator-quality inference from this audit alone.

## Failure modes
Public source/binary mismatch; exceptional/nonfinite kernels; interposition changing dispatch; overflow/underflow and log-exp rounding. These must remain visible in fixtures and source notes.
