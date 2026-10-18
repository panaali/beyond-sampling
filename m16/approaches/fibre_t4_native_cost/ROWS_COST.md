# Dense rows_at numerical expression count

For H hidden units and T inputs, every call executes the same dense array
shapes regardless of the activation mask. With a length-k dot priced as
k multiplications and k-1 additions, the named numerical count is

- `(T-1) H T (2H-1)` for the three/general recurrent dense matrix products;
- `(T-1) H` for adding the input column;
- `T H (2T-1)` for dense A@x at every step;
- `T H` preactivation comparisons;
- `T H T` products in A*g[:,None];
- `T T (2H-1)` for the dense readout matrix product;
- `H-1` for the final boolean reduction under the same declared convention.

These sum to223 at H2T4 and2095 at H8T4. Explicit `t>0` comparisons addT
integer-control units; loop iteration, indexing, assignment/copy and memory
remain separate/unpriced. Diagnostic Ops/X calls are excluded consistently
with the reviewed T121/T146 tracer. Elementary-function/native inner-machine
implementation is not expanded recursively.

The four fixed H2T4 primitive model controls have identical dense shape counts:
892 named units +16 explicit integer comparisons. Their original legacy counts
are34 for allzero and62 each for the other three controls. All numerical
arrays, live flags and legacy counters agree bitwise between original and
instrumented functions. This establishes a local implementation/counting gap,
not a full-estimator ratio; calling frequency and other functions still matter.
