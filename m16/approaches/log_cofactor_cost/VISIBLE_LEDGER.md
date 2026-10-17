# Source-visible numerical ledger (pre-aggregate hypothesis)

Pinned139a7b9e `orthant` only, dimension4. Count scalar add/subtract/multiply/divide, unary sign, absolute value, numerical comparison, sqrt/atan2/log/exp each1. Integer/index/control, mask copying, allocation and memory are separate exclusions. A dense dot of lengthk is k multiplies plus(k-1) additions; a sum overk entries is(k-1) additions. This is an explicitly declared scalar-expression convention, not actual vector/FMA instruction accounting. Treat a square as1 multiply. Exact model-free arithmetic on constants is recorded separately below and may be precomputed under the protocol; no model-dependent executed work is removed.

## Common prefix

The initial inverse is executed even though its covariance is overwritten in the d4 wide branch and unused in the thin branch. Its previously reviewed numeric subset envelope290 is a separate library category, not included in visible expressions.

| Site | Visible count |
|---|---:|
| B @ B.T |112|
| sqrt(diag(S)) |4|
| outer(dg,dg) |16|
| S / outer |16|
| abs(det(Rays)) excluding det |1|
| dt < THIN_DET |1|
| Common visible subtotal |150|

One inverse4 and one determinant4 occur on both branches. Dependency envelopes290/176 yield common category sum616. The branch comparison is included, unlike the first informal arithmetic sketch.

## Thin branch

DU has1000 rows4 columns. Dense DU @ Rays has4000 length4 dots, hence28000 operations. Squaring the4000 entries adds4000; summing each4-entry row adds3000. Squaring1000 q values adds1000, dividing DW adds1000, summing adds999. The final dt division and multiplication add2. Total thin visible beyond common is38001. The model-free denominator `2*pi**2` adds2 operations if charged at runtime. Thus thin category subtotal is38617 excluding those2 model-free constant operations. This is not a complete native/runtime envelope.

## Wide branch

| Site | Visible count excluding listed det calls |
|---|---:|
| 4 cofactor normal rows: determinant sign multiplications16, orientation dots28, orientation comparisons4, orientation sign multiplies16, norms32, divisions16 |112|
| N @ N.T |112|
| 6 pairs: two vector add/subtract and norms24, atan2/multiply/subtract3, min/square/divide3 |180|
| p: six-value sum5, divide1, add1 |7|
| Rt: I-C16, broadcast multiply2048, add2048 |4112|
| max(det(Rt),0) |128|
| 6 pairs: cofactor sign multiplies768; abs/min12; om3840; partial-angle neg/mul/sqrt/atan23072; weighted division/reduction/accumulation4608 |12300|
| final acc division and addition |2|
| Wide visible beyond common |16953|

Cofactor normal rows have16 determinant3 calls;6 batches128 add768, total784 determinant3 calls. Rt adds128 determinant4 calls. There are129 determinant4 calls total including the common prefix. With tentative determinant3 bound104, reviewed determinant4 bound176 and inverse4 bound290, the wide numeric-category subtotal is150+16953+784*104+129*176+290=121633. Each of6 `pi/2` constants adds1; p has8*pi (1; the literal1/16 is folded by Python); final denominator4*pi**2 adds2. Runtime model-free constant overhead is9. Integer powers(-1)**q and(-1)**(k+l), dimension comparisons and index predicates are not numerical model-data operations in this convention and remain explicitly excluded.

These counts are hypotheses for independent source review before corpus contraction. They replace the inherited orthant counters rather than adding to them. Initial inverse/determinant categories, dense products and reductions must not be charged twice. Native library envelopes cover only the stated scalar numeric subset; all other implementation categories remain open. Geometry, triangulation and cold rule generation are absent from this orthant-only ledger, so it cannot establish a full estimator budget or MSE ratio.

Residual native-kernel gaps: the visible matrix products/norms/reductions use the declared scalar-expression convention, while the small determinant/inverse categories use installed lane envelopes. Actual BLAS accumulator initialization/merges, NumPy reduction structure and Python3.12 built-in sum compensation can perform additional work. Thus the combination is explicitly a hybrid source/native subtotal. The5-add six-value reduction is a declared expression charge, not proof the actual built-in sum executes only5 additions. These remaining native interiors prevent a complete as-implemented numeric envelope or a direct sampling-budget conclusion. Replacing these kernels with explicitly implemented scalar loops would be a new estimator variant, not an accounting repair of the pinned source.
