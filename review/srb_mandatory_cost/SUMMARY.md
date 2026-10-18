# Fixed six-budget conditional accounting result

Source/theorem gate0da05327 and driver gate1ae9600a were shared before binding47f447d3 and sole run54502cb1. Independent saved audit363b20d4/shared293cbdde passes all six budgets/four controls/22 progress records and23 pins via independent integer nearest-even rounding and dimension-derived floors. ReviewerCPU .031990211, no author calls. All four integer batch-boundary controls pass; six source-declared budgets are retained. No source estimator/Q/draw_all/model/RNG/native/reference calls.

| Source budget | Kept M | Minimum generated D | Reported FPE | Dense-only FPE floor | Rule5 extra floor | Combined conditional floor | Status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1000 | 584 | 50000 | 2048.062500 | 685.226667 | 1976.640000 | 2661.866667 | conditional_undercharge |
| 4000 | 2348 | 50000 | 8233.500000 | 2754.986667 | 1906.080000 | 4661.066667 | inconclusive |
| 16000 | 9407 | 50000 | 32983.500000 | 11037.546667 | 1623.720000 | 12661.266667 | inconclusive |
| 64000 | 37643 | 50000 | 131987.625000 | 44167.786667 | 494.280000 | 44662.066667 | inconclusive |
| 256000 | 150584 | 200000 | 527985.562500 | 176685.226667 | 1976.640000 | 178661.866667 | inconclusive |
| 1000000 | 588233 | 600000 | 2062493.812500 | 690193.386667 | 470.680000 | 690664.066667 | inconclusive |

Every dense-only comparison is inconclusive. The first combined floor exceeds the reported binary charge by613.8041666666671FPE under the explicit kept-prefix/free-input policy interpretation. The classic-dot alternative has the same conditional decision pattern, but remains a different convention. Exact Fraction values and source float hex, including binary floor/ceil effects, are in results.json. Display rounding is not used for decisions.

The remaining five combined rows are inconclusive, not evidence their charges are complete. Actual extra batches, uncounted arithmetic/comparisons/CDFs/sorting/control or source differences may change the full cost. No historical generated D, prefix tie exclusion, accuracy/MSE outcome, whole-ranking change or shared source adoption is inferred. Model-free setup and original-model advice retain their independent policies.

The arithmetic phase used0.035994353 processCPU seconds, including source parsing/provenance and durable records. It is not estimator timing. Metadata failure/correction: an administrative TASKS update initially looked for a peer active row that had already closed; it made no edit or run and was corrected by anchoring the table header. Scientific artifacts and all numerical outcomes were produced once.
