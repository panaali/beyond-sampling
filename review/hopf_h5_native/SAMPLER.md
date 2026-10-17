# H5 matched named-source convention

Given the true input label for free, one dense H5T4 indicator costs
T[H(2H-1)+H+H+H] + T(2H-1) + (T-1) + 1 = 280:
recurrent matrix-vector, input multiplication, addition, ReLU, final readout,
argmax and integer correctness comparison. The separate analytic initial-zero
implementation removes first recurrent product and addition, saving2H²=50,
so its indicator costs230. This is an explicitly different comparator,
not a claim about unchanged mc_acc's execution.

Hit reduction plus batch accumulation adds N units overall. Named source
loop/index controls add8 per batch for original dense or7 for the analytic
initial-zero form, with2 fixed units. Thus the exact declared budgets are
281N+8ceil(N/1e6)+2 and231N+7ceil(N/1e6)+2. Integer/control items here are only
those exposed by the existing AST convention, not total machine work.

The fixed five-input original-source check uses chunk3, hence two batches,
and predicts1423 explicit units. It uses the frozen dense synthetic H5 model,
including zero/tied inputs, without random draws. True-label sorting is
retained as a separately tagged opaque/source call and excluded under the
free-label comparator convention; no omitted candidate work is justified by
that convention. RNG, memory, runtime, compiler/native equivalence and full
control remain excluded. This control does not measure quadrature accuracy.
