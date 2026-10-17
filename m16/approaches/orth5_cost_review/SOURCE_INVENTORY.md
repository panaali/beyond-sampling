# Source deductions before execution

All counts below follow the pinned source, not a native library-body audit. Rules are thin6/log128. T185 source's proxy uses its own nominal units, and the independent sampler below uses multiply/add/comparison units. They must not be divided without a justified mapping.

## Thin path

One5x5 determinant, one1296x5 by5x5 matrix product, elementwise square of1296x5 values, a row reduction of length5 for each node,1296 powers to2.5,1296 weight divisions and a final1296-term sum. Absolute determinant and threshold comparison precede the branch. Scalar prefactor operations follow; constant-folded literals are not runtime work. Proxy is51,965.

The unpadded mathematical matrix product alone has32,400 multiplies and25,920 additions (58,320 scalar operations). This is a mathematical expression count, not a proven installed-kernel lower/upper: vector padding, dispatch, fused operations and zero simplification are separate. Noninteger powers remain opaque. The thin fixed-rule numerical failures on wide product cones are retained; counting does not repair them.

## Log path

- One initial5x5 determinant.
- For five normal rows,25 scalar4x4 determinants, five length5 dots for orientation, five length5 norms and vector orientation/scaling.
- One5x5 self-transpose product for the correlation matrix.
- Ten pairs:20 length5 norms,10 scalar atan2 calls and source endpoint factors.
- A128x5x5 affine matrix stack.
- Five batches of128 4x4 determinants, and30 batches of128 3x3 determinants. Together with the normal construction:665 determinant4 and3840 determinant3 scalar instances.
- Ten correlation pairs:30 vector atan2 calls, each128 elements;30 vector square-root calls for conditional angles and10 for the final denominator;10 length128 final sums. Array products, sums, maxima, scalar clamps and final prefactors are additional.

Proxy is173,060. No determinant5 price exists in this lane. Earlier determinant3/4 and NN/SYRK prices require actual dimensions, strides, backend and call-path applicability; a source function name alone is insufficient. Native and source quantities cannot simply be added if already included in another category.

## Sampler convention

For H8,T5, zero initial hidden state, a direct scalar forward with its known-zero first recurrent product omitted has336 multiplies,291 additions and45 comparisons:672 total. Multiplications:40 input scalings+256 recurrent+40 readout. Additions:224 recurrent dot+32 input/recurrent combinations+35 readout. Comparisons:40 ReLU+4 argmax+1 correctness. Supplying the true second index is model-free here, not counted as network work. Excluding the final integer correctness comparison gives671 numeric-only units.

Executing the first zero-state recurrent product adds64 multiplications and64 additions, giving400+355+45=800, or799 numeric-only. This is not a claim that batched NumPy/OpenBLAS executes exactly those scalar operations. RNG, rank computation, allocation, loop/index/control, memory and batch/setup work remain separate. No sampling has been performed for this inventory.

## Instrumentation confound preserved

Initial source adapter b39475b9 would have wrapped literal arithmetic, preventing some CPython constant folding. This was identified by source review before any run. Revision ba5b24f0 leaves compiled numeric-constant subtrees unchanged and adds explicit controls. No results were overwritten or selectively rerun. Source-expression counts still do not expose compiled NumPy/libm internals.
