# Frozen expression convention and closed-form checks

A multiplication, addition/subtraction, numerical comparison, unary sign,
elementary function and remainder each count one numerical unit. Dense dot of
length k counts k multiplications plus k−1 additions. These are source-expression
units, not processor fused instructions or libm internals. Boolean elementwise
operators retain distinct tags; scalar integer arithmetic/control is separated.
Integer-array comparisons currently retain the array tag and are included in
the inclusive total, not silently discarded. Memory allocation/gather/repeat,
index generation and Python loop/dispatch are excluded with a visible census.

Opaque lexsort, argsort and searchsorted calls retain original operands/results,
axis/shape signatures and multiplicities. This initial trace cannot establish
full numerical cost while these comparisons remain unpriced. A future closure
must identify actual native paths or explicitly freeze a replacement; observing
a Python comparison-sort trace would not price the original NumPy implementation.

## Fixed native-shape arithmetic

For n incoming intervals and H hidden units, recurrence einsum has2nH outputs
of lengthH:2nH(2H−1). The broadcast input products are2H, followed by2nH
additions. Sum4nH²+2H, including actual time-zero dense work.

For m=nk root coefficients, atan2 costs m; the two scalar pi/2 divisions cost2;
two shifts and two remainder arrays cost4m; strict lower/upper tests cost4m;
coefficient-zero tests cost2m. Thus numerical arithmetic/comparisons are11m+2,
not counting Boolean OR m and Boolean AND4m. Original repeat versus tile is
memory-only in this convention, but their incorrect/correct outputs can change
all downstream dimensions. Both variants remain separate.

For n post-split intervals, midpoints cost2n, sin/cos2n, gate evaluation3nH,
gate comparisons nH and applying the gate to both coefficients2nH. Total
n(4+6H). Readout einsum with T classes costs2nT(2H−1).

## Actual dense sampling comparator

Each sample's T recurrent steps costT[H(2H−1)+H+H+H]
=2TH²+2TH. Readout addsT(2H−1), argmax T−1 and one correctness comparison.
Total2TH²+4TH; H4T4 gives192. Omitting the known zero-state recurrence at
step0 gives160, but that optimized source is not what zoo44.mc_acc executes.
These are prediction/indicator units conditional on a supplied true label.

The actual m-sample Boolean hit reduction adds max(m−1,0) additions, plus
one accumulator update per batch. For positive batches totalling N inputs,
these add N units, in addition to the192N indicators at H4T4. Final accuracy
division adds one. Remaining explicit batch-size/loop integer expressions are
separate. Input generation and true-label sorting have distinct exclusions;
no fresh random reference is required to validate forward semantics.

## Failure discriminators

1. If the arithmetic observer changes outputs/legacy counters, preserve failure
and repair only the observer before a trained trace. Original/tile arms compare
to their respective baseline, never to each other as a parity requirement.
2. If root/dense site formulas disagree, inspect broadcast dimensions and source
constant expressions, not a fitted multiplier. Scope follows saved actual sites.
3. If opaque sorts dominate possible cost headroom, source/dispatch analysis or
a frozen replacement is necessary; wall time is not an operation count.
