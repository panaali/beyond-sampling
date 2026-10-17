# Operation difference, not a new total allowance

Use the existing scalar convention: add/subtract/multiply/divide/comparison cost1, square root cost20; integer squares are scalar multiplies. A length-six dot/norm-squared costs11 operations. This is a symbolic implementation model, distinct from wall/CPU and vector-library overhead.

The old per-triangle frame block has:

| Block | Scalar charge |
|---|---:|
| Two length6 edge differences | 12 |
| First edge norm | 11+20 |
| First normalization plus zero-norm condition | 6+1 |
| Projection of second edge | 11 |
| Subtraction of first projection | 12 |
| Second edge norm | 11+20 |
| Second normalization plus zero-norm condition | 6+1 |
| Total | 111 |

The new frame computes these operations once per cone, then scales lu, lv and v1 at each outer node (3 multiplies). Therefore the unchanged common algebra saves **108*n−111** operations for n inner triangles, before bookkeeping. Input-validation, array construction, endpoint projection, ray normalization, all trigonometry, screens and fallback are retained. Matrix-vector versus einsum projection has the same scalar dot count, but different allocation/library behavior.

This difference is small relative to T52's inherited conservative10000 operations per analytic triangle. A new tighter allowance would be an accounting improvement rather than a physical saving and must be reported separately. We do not convert that allowance into a claimed5x or10x algorithmic speedup. The measured runtime comparison includes fixed-frame preparation and endpoint generation on each call; no per-model preparation is free. Imports/build and repeated model geometry are not part of this primitive comparison.

The first benchmark is ordinary same-cone geometry, where neither engine falls back. Trained cones with many positive fallbacks may benefit less. No full end-to-end estimator cost or accuracy-versus-sampling result is claimed by this memo.
