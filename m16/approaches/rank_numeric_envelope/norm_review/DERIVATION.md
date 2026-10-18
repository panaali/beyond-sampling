# Installed DNRM2 numeric subset

The pinned wrapper returns zero for N<=0, uses one absolute value for N=1,
and dispatches positive N>1/nonzero-stride calls through gotoblas+0x348.
The observed target is dnrm2_k_SKYLAKEX. All five local architecture targets
(PRESCOTT, NEHALEM, SANDYBRIDGE, HASWELL, SKYLAKEX) have the same floating
instruction sequence and loop structure; older targets add a prefetch/alignment
padding. This reduces sensitivity to the historical architecture choice but does
not establish historical binary/binding provenance by itself.

For positive N>1 and positive stride, each group of eight loads eight inputs,
squares each, and adds each to one of four x87 accumulators. The remainder
adds each square to accumulator zero. Both stride paths execute N multiplies
and N additions, followed by three accumulator reductions and one square root.
Thus the named numeric subset is 2N+4, with no floating comparisons. N=1 is
handled before dispatch and costs one named absolute value, not this expression.
The four accumulators receive index modulo four within full eight-input groups;
remaining inputs all enter accumulator zero. Final reduction is
((s0+s2)+s1)+s3. All arithmetic stays on the x87 stack until the binary64 return.

This is an installed instruction count in named scalar units, including x87
extended arithmetic and a square root as one unit. It is not a cycle, energy,
or bit-complexity count. Loads/stores, casts, integer guards/indexing, branch
and dispatch work, exception handling, and runtime costs remain unpriced.
No parent routine is charged here. No abstract geometry norm is subtracted
without demonstrated callsite attribution. The saved census supplies finite
calls including one cold setup; it does not prove a universal call bound.

Alternatives: an unseen architecture could implement another norm (inspect all
installed targets, not just the current name); a naive binary64 translation
could accidentally agree on ordinary values while misrepresenting extreme
values (freeze tiny/huge/mixed/nonfinite controls and emulate longdouble order).
Finite control parity supports the implementation reading, not all-input
correctness or historical dispatch coverage.
