### T431 results (HP5, 100 sealed retrains; `python3 t431_analyse.py` → `results/hp5_t431.json`)
| run | C mean (FPE) | cost ≥ B/2 | **F** (truth var subtracted) | F raw | T281 ratio | bias pp | RMS pp | truth share | strict (CI upper < 1) | lenient (CI upper < 10) |
|---|---|---|---|---|---|---|---|---|---|---|
| T10 4e6 (PRIMARY) | 3,997,507 | yes | 0.540 [0.343, 0.762] | 0.580 [0.383, 0.802] | 0.558 | +0.0007 | 0.0104 | 0.069 | PASS | PASS |
| T10 1e6 | 999,742 | yes | 0.406 [0.270, 0.575] | 0.416 [0.280, 0.585] | 0.412 | +0.0002 | 0.0176 | 0.024 | PASS | PASS |
| isg 1e6 | 1,001,623 | yes | 0.627 [0.422, 0.851] | 0.637 [0.432, 0.861] | 0.630 | +0.0040 | 0.0217 | 0.016 | PASS | PASS |
| control 1e+06 | 1,000,000 | yes | 0.806 [0.597, 1.038] | 0.816 [0.607, 1.048] | 0.808 | +0.0004 | 0.0246 | 0.012 | fail | PASS |
| control 4e+06 | 4,000,000 | yes | 1.015 [0.714, 1.377] | 1.055 [0.754, 1.417] | 1.015 | -0.0019 | 0.0140 | 0.038 | fail | PASS |

Sensitivity: the recipes' FPE unit is T(2H²+2H) = 5,440 FLOPs; charging C as FLOPs/5,948 (×0.915) instead:
- T10 4e6 (PRIMARY): F 0.494 [0.313, 0.697]
- T10 1e6: F 0.372 [0.247, 0.526]
- isg 1e6: F 0.573 [0.386, 0.778]
