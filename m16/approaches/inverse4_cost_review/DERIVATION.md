# Scoped floating-expression envelope

The installed double-precision NumPy inverse calls GESV with a 4×4 identity right-hand side. Its one-thread small path is `dgetrf_single → dgetf2_k`, followed by `dgetrs_N_single` and packed lower/upper triangular solves. The runtime dispatch table selects SkylakeX kernels, GEMM unroll M=16,N=2; the 4-row remainder and two RHS blocks of width2 execute scalar solve tails. No GEMM is called in the frozen controls. Binary hashes and table addresses are in environment.json/runtime_dispatch.json. This is an arithmetic-expression envelope, not instruction or hardware-cycle accounting.

Count a scalar +, −, × or division as one, an FMA as two. Assignment, absolute value, comparison, integer/index/control expressions, memory traffic and allocation are excluded here and must be charged separately for any complete implementation comparison.

| Piece | Count | Reason |
|---|---:|---|
| Strided DOT plus outer subtraction | 14 | Lengths1,1,2 use 2n+1 (including final accumulator sum), plus one `b[i] -=` each |
| GEMV tails | 32 | Nonempty (m,n)=(3,1),(2,2),(1,3), each 2mn+2m; (4,0) returns without FP arithmetic |
| LU pivot reciprocals and scaling | ≤10 | At most4 normal nonzero-pivot divisions and3+2+1 scaling multiplies |
| LU subtotal | ≤56 | Left-looking factorization, not textbook right-looking elimination |
| Lower plus upper solve | 128 | Four (4×2) calls, each8 diagonal multiplies plus24 update arithmetic; unit-lower multiplication is retained |
| Upper diagonal packing | 4 | One reciprocal for each upper diagonal; lower unit packing writes1 |
| Total | ≤188 | Floating-expression category only |

The normal finite nonsingular short path has this expression budget. Zero, singular and subnormal branches can skip arithmetic; pivot swaps copy data. The saved zero/duplicate/subnormal fixture envelopes are50/56/185; the two-matrix batch is376. This is not a guarantee of numerical accuracy for subnormal, overflow or ill-conditioned matrices. NumPy error handling and floating-point status management remain outside the arithmetic category.

The public v0.3.34 files are not proven identical to the installed 0.3.34.106.0 source tree. The local small-path disassemblies establish dispatch and corresponding scalar-tail structure independently; full large-size branch source equivalence is unnecessary here and unclaimed. `runtime_dot_compute_disassembly.txt` isolates ELF0xa3d440..0xa3d790 because multiple architecture implementations reuse this static symbol name. Runtime transparent wrappers verify all actual shape arguments, restoring the process-local dispatch table in finally. They do not modify installed files or other processes.

## Composition and unresolved work

The T146 exact census has2,659,906 inverse calls: at most500,062,328 units in this FP category. T153 recipe has2,138,536 calls:402,044,768. Parent visible totals plus this subset are1,053,463,096 and858,021,483, respectively. Those sums are **not full-cost upper bounds**: inverse integer/control/absolute-value/comparison and packing-copy work remains unpriced, as do parent SVD/rank, determinant and other opaque categories. Do not interpret an upper bound on one subset as either a complete upper or lower bound for the whole implementation, or infer a matched-budget MSE result.

Potential next complete-count methods include separately instrumenting small-path C integer/control expressions, or deriving conservative local binary branch bounds. Allocator/library/runtime work needs a consistent project-wide inclusion convention; timing alone cannot supply an operation bound. No claim excludes such approaches.
