# r1_v1b2 (T313): R1W V1 with exact B2 prefix sharing and instrumented FLOPs (cost only; estimates unchanged)

↑ [r1_walk (T306 V1)](../r1_walk/README.md) · [r1_b2b5 (T307, B2 measured)](../r1_b2b5/README.md) · TASKS Directive 8
cost convention

## What changed
**Only the cost.** `v1b2.est(model)` returns the estimate of frozen V1 (`r1_walk/r1w2.est`, same code path). The
cost is `v1b2.cost(model)`, instrumented under the written convention:
- Every weight-dependent operation is counted as implemented: the walk step-evaluation (value + tangent +
  gate-flip roots) is 1,218 FLOPs; readout and correct-set solve per cell is 790 FLOPs.
- Phi at weight-dependent points is 20 FLOPs (+3 for the ^8 on the g23 line).
- Model-free tables are free: GL nodes and weights, phi at the nodes, transverse points, the v4 node, Phi at
  model-free points (v2 + tmax, infinity, the blind "g12 correct" profile).
- **B2:** in the first cell of each walked line all rows are at the same t, so step-evaluations are charged per
  distinct (input, tangent) prefix across the 720 triples. Later cells are charged per row from the restart step.

## Results (`python check.py`, `results.json`; all 80 DEV2)
- **Estimates: max |V1-B2 - V1| = 0.0** on all 80 (<= 1e-12 required).
- **Cost:** mean **6.53e7 FLOPs** = 10,971 FPE at 5,948 FLOPs/FPE (11,329 at 5,760), against frozen V1's charged 7.90e7
  FLOPs: **-17%**. Range ~5.9e7-7.6e7 per model.
- [inferred projection] HP2's per-model errors are unchanged, so F scales with C: about 16.35 x (10.97 / 13.3) ≈
  **13.5**. That still fails the lenient bar, which needs C <= ~8.1k FPE at RMS .99. The T281 scorer recomputes the
  exact HP2 cost per sealed model with `v1b2.cost`.
- **No new HP2 submission:** the estimates are identical to the already-scored V1.
