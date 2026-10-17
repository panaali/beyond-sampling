# DLARFG exclusive source envelope

Pinned source is copied locally and hashed in protocol.json. For N<=1 it returns after a dimension guard with TAU=0: no floating numeric work under this convention. For N>1, one initial DNRM2 is called and XNORM compared to zero. The zero branch then returns. The other branch computes BETA and may rescale small values. KNT starts0; each loop pass increments it and the back edge requires KNT<20. Hence at most20 scale passes and20 final unscale multiplications, independently of a convergence or nonzero-input assumption.

Exclude child internals and integer/index/control/memory work from this subtotal. Treat SIGN/copysign as one explicitly named primitive, without claiming its internal instruction price.

| Numeric source category | Uniform N>1 upper |
|---|---:|
| arithmetic |66|
| negation |2|
| SIGN/copysign |2|
| absolute value |21|
| numeric comparison |22|

Arithmetic is1 SAFMIN division+1 RSAFMN division+40 BETA/ALPHA loop multiplications+2 TAU operations+2 final DSCAL-factor operations+20 BETA restorations. Two BETA constructions give two SIGN and two negations. One outer and20 loop ABS/threshold comparisons plus the initial XNORM==0 produce21 ABS and22 numeric comparisons. Total113 named primitives. Early/zero/no-rescale branches reduce work.

Additional child-call envelopes, whose numeric work is NOT included in113: at most2 DNRM2(N−1,INCX),2 DLAPY2,2 DLAMCH, and21 DSCAL(N−1,INCX). These are per-call upper bounds, not observations. The global DNRM2 census has no parent attribution. Geometry norm overlap is possible, not established: NumPy may evaluate those norms through dot/sqrt without calling this symbol. Do not treat the census as rank-exclusive or subtract AST norm work merely from a shared operation name. One can intentionally double count to form a conservative upper bound only when every constituent is itself a valid envelope and the duplication is declared.

Frozen160 controls cover N1/2/4/24, strides1/3, zero/positive/alternating/subnormal X, alpha0/±1/±1e−310. The source adapter calls the same native children and compares with native top-level DLARFG, so it tests exclusive source/control parity, not independent child arithmetic. All failures are retained. N<=1 and zero-norm cases matter; division or norm work must not be charged as observed when the source returned early. Final source-to-installed equivalence beyond fixtures remains qualified.
