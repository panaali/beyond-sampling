# Explicit inherited sampler convention

For three scalar inputs, H hidden ReLU coordinates, three output logits, and an initial zero hidden state, omit the known-zero first recurrent matrix product. With each multiply/add/comparison charged once:

| Category | Count |
|---|---:|
| Multiplications | 2H²+6H |
| Additions | 2H²+3H−3 |
| ReLU and output-argmax comparisons | 3H+2 |
| Named numeric total | 4H²+12H−1 |

The two nonzero recurrent steps contribute2H² multiplications and2(H²−H) additions. Three input-vector products contribute3H multiplications; adding them at the last two steps contributes2H additions. Readout contributes3H multiplications and3(H−1) additions. Three ReLU layers and a three-way argmax contribute3H+2 comparisons.

Thus H4/5/6 use111/159/215 numeric units. This is the existing T172 convention, not a new measured native kernel. Model-free true-label ranking, RNG/input generation and final integer correctness comparison are excluded consistently; adding the final integer comparison would define a different denominator. The older saved sampler114/162/218 includes three true-label comparisons and excludes final correctness. Preserve it rather than silently replacing historical ratios.

Dense scalar expression counts and unit elementary-function invocations for the estimator are an abstract source convention. Memory/control, SIMD inactive lanes, batching, native library internals and model-free setup are not fully priced. No processor/runtime-independent speed claim follows from these totals. Per-model sample budget uses floor(estimator named numeric count / comparator units); a mean-cost allocation and individual budgets are distinct comparisons.
