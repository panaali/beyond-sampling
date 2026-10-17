# Pinned3x3 determinant category envelope

This is a dimension-specific hypothesis pending transparent controls, not an unreviewed rescaling of T1584x4. Exact installed binary hashes are pinned in frozen/det_environment.json and match prior T150/T157/T158. The public NumPy source and disassembly show determinant reconstruction through GETRF, diagonal log summation and exp/sign multiplication, not a three-diagonal product.

Exported GETRF computes the matrix-size product and dispatches the small path for3*3<=9999. The existing dgetrf_single code computes a block width from min(m,n)/2 and the runtime unroll2; for min3, its small-block threshold reaches dgetf2_k. The captured runtime table uses the same SkylakeX child kernels. Transparent controls must check dimensions3/lda3, IAMAX lengths3,2,1, sole strided DOT length1 with strides3/1, GEMV shapes(3,0),(2,1),(1,2), and at most SCAL lengths2/1. No solve, GEMM or packed triangular kernel is allowed. Singular/subnormal conditions can skip later arithmetic; preserve those paths.

| Category | Bound | Derivation |
|---|---:|---|
| Strided DOT and outer subtraction |4|One length1 strided DOT: the inherited scalar-tail formula2n+1 gives3, then outer subtraction1. This deliberately includes the accumulator merge.|
| GEMV scalar tails |14|Shapes(2,1),(1,2):2mn+2m=8+6; empty(3,0) returns.|
| Pivot reciprocals and scaling |6|At most3 divisions and2+1 scalar multiplies.|
| LU floating arithmetic |24|No triangular inverse solves.|
| IAMAX absolute lanes |18|n3,2,1 inherited independent lane bounds8+8+2.|
| IAMAX MAX/comparison lanes |29|11+11+7.|
| Pivot absolute/comparison lanes |12|Three pivots, at most2 absolute lanes and2 comparisons each.|
| SCAL numeric guards |4|At most2 comparisons for each of2 calls.|
| Numeric guard subtotal |63|Includes irrelevant packed lanes conservatively.|
| Reconstruction add/multiply |4|Three log accumulations, final sign multiplication.|
| Reconstruction sign negations |6|At most two per negative diagonal.|
| Reconstruction comparisons |3|One diagonal sign comparison per successful pivot.|
| log / exp calls |4|At most3 log and1 exp; singular skips logs but retains exp.|
| Reconstruction subtotal |17|Elementary functions counted as named calls, internals excluded.|
| Numeric-category total |104|24+63+17.|

The DOT row's3 operations are1 multiplication plus1 accumulation plus1 merge, not a mathematical minimal1-operation dot. The loop-branch-independent upper allows exceptional and unused lanes, under the same masking/runtime assumptions as T157. It is a source/scalar-lane convention, not cycle or instruction pricing. Three pivot inequalities, three integer sign accumulations, remainder/parity/info tests are distinct integer/control categories; no silent mixing with numeric-only sampling. Other dimensions, backend dispatch and threading are outside this proof. This subset bound cannot itself certify either a full estimator upper bound or lower bound.
