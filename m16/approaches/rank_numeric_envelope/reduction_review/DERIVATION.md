# Reduction source envelope, partial

Only pinned DGEQR2, DGELQ2 and DGEBD2 parents are covered here. Their exclusive
operations are copies/constant assignment and integer/index/control expressions;
there is no named scalar floating arithmetic outside the DLARFG and DLARF1F
children. This does not mean zero runtime, and does not cover DGEQRF/DGELQF,
DGEBRD, GESDD or any other driver. All observed DLARFG work is already priced in
the earlier reflector ledger and must not be added again.

Installed reducers use unprefixed dlarf1f_, with ILP64 dimensions, as shown by the
direct calls and argument loads. The initially empty scipy_dlarf1f_64_ probe is
retained alongside the corrected actual-symbol disassembly.

For one application let L be full reflector dimension and C the other matrix
dimension. LASTV<=L, LASTC<=C. Numeric parent/scan source units are: one TAU zero
comparison, at most L-1 V zero comparisons, matrix scan, and at most two -TAU
negations (or one 1-TAU subtraction for LASTV=1). ILADLC(L,C) uses at most2+LC
comparisons, including the two corner probes. ILADLR(C,L) uses at most2+L(C+1),
because each column can recheck row1 when I=0. Thus left bound L+4+LC and right
bound2L+4+LC. This deliberately counts potential source comparisons optimized
away by compiled short circuit. Empty/allzero cases can exit earlier.

The first proposed uniform bound transposed the ILADLR scan dimensions and was
caught independently before execution. scan_bound_correction.json preserves it.
The controls explicitly cover right application L>C and allzero matrices.

Shape contraction follows the literal reducer loops. QR applies left on
(m-i+1,n-i) when i<n; LQ right on(m-i,n-i+1) when i<m. Upper-bidiagonal reduction
has left(m-i+1,n-i) and right(m-i,n-i) for i<n. Lower-bidiagonal reduction has
right(m-i,n-i+1) and left(m-i,n-i) for i<m. m,n are actual saved kernel dimensions,
not arbitrary original matrix dimensions. Every diagnostic cold phase is equal;
charge one cold phase plus eight model chunks.

BLAS child pricing remains OPEN. Current installed target pins: DAXPY slot388,
DSCAL390, GEMV N/T3a0/3a8, GER3b0, all SKYLAKEX. Their actual lane/reduction work,
wrappers and dispatch must be priced separately. Shared-native-child parity
controls validate application wrapper semantics only, not independent BLAS
correctness. All limits/binding/finite-precision qualifications remain.
