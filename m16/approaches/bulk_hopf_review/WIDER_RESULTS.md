# Wider saved Hopf evidence audit

Frozen claim55b629e6/protocol689b0768 covers75 already saved H4/H5/H6 T4 cases. Every prior H4/H5 row and the prior20 H6 rows is unchanged; only H6 seed4 adds five rows. All45 clean/noise/pooled groups are contracted exactly from saved binary float values before display rounding. There are no new trained-model, reference or MC evaluations.

The following pooled ratios retain native96/140/192 sampler units and unaudited native Hopf counts. The adjusted column subtracts the saved MC variance estimate and clamps at zero; zero is not a zero-bias certificate. The h128 column treats a finite quadrature reference as a comparator, not certified truth. The last column uses one nominal95% simultaneous family of75 marginal binomial truth boxes, allowing shared MC input streams. Beta inversions are ordinary numerical evaluations, not outward-rounded enclosures.

| H | Seed | Raw MC ratio | Adjusted ratio | h128 ratio | Truth-box ratio interval |
|---|---|---|---|---|---|
| 4 | 0 | 0.103897 | 0.037066 | 0.038340 | [0.000000, 1.385944] |
| 4 | 1 | 0.019970 | 0.000000 | 0.006489 | [0.000000, 0.967668] |
| 4 | 2 | 0.055307 | 0.000000 | 0.005640 | [0.000000, 1.128090] |
| 4 | 3 | 0.058806 | 0.000000 | 0.019063 | [0.000000, 1.241766] |
| 4 | 4 | 0.025306 | 0.000000 | 0.012470 | [0.000000, 0.986083] |
| 5 | 0 | 0.049309 | 0.000000 | 0.012111 | [0.000000, 1.108238] |
| 5 | 1 | 0.413702 | 0.348023 | 0.109038 | [0.021139, 2.066768] |
| 5 | 2 | 0.114971 | 0.044926 | 0.020306 | [0.000000, 1.442471] |
| 5 | 3 | 0.066084 | 0.000426 | 0.037535 | [0.000000, 1.202031] |
| 5 | 4 | 0.299651 | 0.232057 | 0.101996 | [0.000000, 1.951437] |
| 6 | 0 | 0.293274 | 0.222880 | 0.047140 | [0.000030, 1.991347] |
| 6 | 1 | 0.108395 | 0.040516 | 0.034259 | [0.000000, 1.436002] |
| 6 | 2 | 0.546745 | 0.478642 | 0.109149 | [0.039353, 2.399916] |
| 6 | 3 | 0.040199 | 0.000000 | 0.015438 | [0.000000, 1.111468] |
| 6 | 4 | 0.493896 | 0.427170 | 0.102482 | [0.097543, 2.078308] |

All15 pooled point estimates are below1, but only H4 seeds1 and4 have simultaneous upper bounds below1 under this native convention. The others are inconclusive. No displayed lower bound exceeds1. This wider75-case family is deliberately more conservative than the earlier25-case H4 family; both are retained.

Separating clean cases exposes raw point ratios above1 at H5 seed1 (1.487255), H6 seed2 (1.996334), and H6 seed4 (3.031155). Their simultaneous lower bounds are .123538, .275926, and .737645, respectively, so none establishes a failure. All15 noise-only raw group ratios are below1; only H6 seed3 has a noise-only simultaneous upper bound below1 (.970728). These are fixed finite groups of four noisy cases, not a noise-distribution guarantee.

Uniform allocation compares the mean squared error with native_sample * mean(p*(1-p)) / mean(native_ops). Matched allocation uses mean(native_sample * p*(1-p)/native_ops). The exact rational calculations preserve both; the largest relative raw-ratio difference over45 groups is .001761719. Pooling and individual budget allocation are not interchangeable, even when the numerical difference here is small.

The original source hash37e5cb781c74b7c030ac305b5836bc9938963fe5fc56c07eaf1d56ee8d3422fc is unchanged. The earlier mixed-zero root defect and universal-continuity counterexample remain separately valid; no saved trained case is shown to trigger either. T183 owns native-cost closure. None of these finite-table checks proves quadrature accuracy, full-cost superiority or advancement to a wider challenge rung.

Arithmetic CPU 0.012198615 seconds; no model calls. Independent arithmetic review pending.
