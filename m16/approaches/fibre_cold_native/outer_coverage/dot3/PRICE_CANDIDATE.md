# Length3 numeric-path candidate, not yet a corpus replacement

All90 frozen controls preserve64bit outputs and restore the original slot. Norm30controls always reach one n3/inc1/inc1 native DOT, including strided/reversed inputs copied by norm. Dot contiguous/reversed20controls and matmul contiguous10 also use that path. Dot/matmul positive-stride2 controls20 use n3/inc2/inc2. The10 reversed matmul controls make no native DOT call; source routes their nonpositive stride through NumPy's scalar DOUBLE_dot fallback. No empty event is counted as free.

For the pinned binary:
- Contiguous n3 DOT takes the odd tail FMA at a3d498 and two-FMA pair at a3d4c6/a3d4d3:6 floating arithmetic units. DOUBLE_dot adds native result to its initialized zero at348e3:1 more. Candidate dot7; default scalar norm adds its named sqrt1, candidate norm8.
- Positive nonunit strides with n3 bypass the four-element block, initialize two accumulators, run three scalar FMAs at a3d5ea (6units), and sum accumulators at a3d5f7 (1). The same NumPy final add gives8.
- NumPy nonpositive/unsupported-stride DOUBLE_dot fallback at34880..34899 performs three scalar products and three adds:6. No BLAS event is expected.

A zero/nonfinite value does not change these fixed arithmetic paths. FMA is counted as one multiply plus one add, matching prior named arithmetic convention. Initialization masks, copies, index/loop/branch control, Python/ufunc dispatch, allocation/runtime, floating status handling and probe overhead are excluded. Norm's sqrt is retained as one named elementary operation; this does not price every native ufunc instruction.

These are finite selected-path candidates with static source generalization still requiring independent review. The saved trained ledger contains dimensions, not dtype/layout histories. No new inherited1,838,695,310 replacement has been made. Norm inputs and explicit dot arrays require a separate pinned source dtype/layout applicability argument; other matrix operations remain unpriced. The10 reversed matmul paths are source-qualified until installed parent branch correspondence is checked. Do not infer sampler/runtime wins.

Independent saved dispatch/path review920a9da1 now passes all90 records and stated numeric subsets. A separately published/gated source-only applicability contraction is in applicability/: selected ten-site upper2,103,777,710 conditional hybrid; final saved contraction audit pending. Original candidate/control artifacts remain unchanged.
