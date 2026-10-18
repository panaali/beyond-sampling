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

## T281 scorer re-count under the written convention, without B2 (claude/hp1-scorer 78a722c9, cost_audit.py)
- R1W V1, HP2: C 13.3k -> **12.1k FPE**, F 16.35 -> **14.92 [10.74, 19.38]**. Still a lenient fail.
- T300, HP2: C 1,697 -> 1,551 FPE, F 9.72 -> 8.89 [6.42, 11.86]. Point pass; the upper CI is > 10.
- HP1 clean: R1W 8.30, T300 5.40.
- **Differences from v1b2.cost:**
  - B2 is not applied there (about -2.6k FPE of first-cell sharing).
  - The scorer keeps 5,760 FLOPs per value+tangent forward-equivalent: 1,152 per step-evaluation, vs my
    instrumented 1,218 + 790 per cell.
  - The per-node combination is charged 4 per (triple, node, region) in both counts.
- With B2 the expected HP2 C is ~10.9k FPE, so F ≈ 13.4: still above 10.

## T281 scorer with v1b2.cost (claude/hp1-scorer 7dea5111; v1b2.py sha256 a09b395d…; estimates asserted identical)

| cost count | HP2 C (FPE) | HP2 F [95% CI] |
|---|---|---|
| v1b2.cost (B2 sharing) | 10,962 | **13.51 [9.72, 17.55]** |
| same count, B2 disabled (what frozen walk_line executes) | 13,656 | 16.83 [12.11, 21.86] |
| scorer's own count under the written convention | ~12.1k | 14.92 |

- HP1 clean: 7.46.
- **Caveat (scorer's flag, which I agree with):** B2 is a saving that the frozen code does *not execute*. walk_line runs
  every row through all T steps in the first cell. Charging B2 is legitimate only for an implementation that
  actually evaluates the shared prefix tree once. That implementation was not written, so the 10,962 figure is a
  cost *for a prefix-tree implementation*, not for the frozen code.
- **The verdict is a lenient fail under every count.**
