# Numerical categories on the pinned small inverse path

This follows T150 and counts a numerical absolute-value, comparison or MAX per executed scalar lane. Packed operations have two lanes even when a lane is duplicated or zero. The result deliberately overcharges irrelevant upper lanes. Bitmask setup, shuffles, integer arithmetic/indexing/control, pointer checks, memory and floating-status handling remain outside this category. It is an expression/lane convention, not CPU-instruction accounting.

## Pivot search

The installed `idamax_k_SKYLAKEX` receives n=4,3,2,1 and inc=1. It first takes the absolute value of the first entry in a two-lane register (two charged lanes), duplicates it, and initializes four accumulators. Since n−1≤3 it jumps directly to the short unaligned path at ELF0xa3a018, bypassing all long loops and alignment-dependent vector bodies.

The remaining pair, if present, has two abs and two MAX lanes. The remaining single, if present, is duplicated before two abs and two MAX lanes. Reduction always uses three packed MAX operations plus one scalar MAX: seven numeric comparison lanes. Index search rereads four scalar values for n4 (eight abs lanes), or two for n2/n3 (four abs lanes), and performs up to4/2 scalar comparisons. The final unpaired index is returned without another comparison. n1 performs no index search.

| n | abs lanes upper | MAX/comparison lanes upper |
|---|---:|---:|
|4|14|15|
|3|8|11|
|2|8|11|
|1|2|7|
|sum|32|44|

MAX selects its second operand for equality/unordered inputs. COMISD followed by JE accepts equality or unordered status. `controls.py` emulates these actual semantics; it does not assume mathematical/NumPy argmax behavior. All4680 vectors over the fixed eight-element alphabet match the installed direct function. The comparator path bounds do not require a finite maximum. Signaling-NaN exception behavior is not modeled; the saved NaNs are quiet and the project runtime masks invalid traps. The numeric lane bound is unchanged, but no general exception-safety guarantee follows.

## Other numerical guards

The four LU pivot tests each perform `pivot != 0`, and for nonzero/unordered input execute a two-lane abs mask plus one comparison with DBL_MIN (ELF0x427779..0x427799): at most8 abs+8 comparisons. Zero, NaN and subnormal branches skip parts of subsequent arithmetic; the fixed pointer clamp is integer/control and remains outside this table.

At most three SCAL calls occur. Inspection of `dscal_k_SKYLAKEX` gives at most two scalar zero comparisons per call; use6 conservatively. For the captured arguments the first special branch is skipped and the count can be lower. Each triangular driver has at most two numeric beta comparisons; use4 total, retaining the source's second comparison even when the installed beta=1 path eliminates it. Driver scaling is not reachable for the fixed GESV beta, consistent with no GEMM/beta call in observed controls. The other reached small kernels have no additional numerical comparison or absolute-value operations; dimensions, indices, loops and strides remain integer categories.

Thus added numerical categories are32+44+8+8+6+4=102. Together with T150188 arithmetic units,290 is a conservative upper bound on this numeric subset for the pinned single-thread4×4 path. Eight fixed inverse controls preserve uninstrumented results/exceptions under transparent dispatch capture, including nonfinite and subnormal inputs; this checks branch plumbing, not numerical accuracy.

## Interpretation

Composing parent counts is valid only as a category subtotal. Exact2,659,906 calls contribute≤771,372,740 numerical units; recipe2,138,536 calls contribute≤620,175,440. Parent visible totals plus these are1,324,773,508 /1,076,152,155. Neither is a full implementation upper bound or lower bound, because a conservatively bounded subset is combined with still-unpriced work. No sampling advantage or challenge solution is established here.
