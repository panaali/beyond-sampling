# Fixed retained-panel result — independently audited

The one predeclared H16T4 zoo0 chamber panel contains 51,552 retained rows. All error and cost values pass the frozen validity checks. Raw3660020e was produced once after source reviewbd31c4b9/shared1a52d40f and immutablegated12629f0; one exact frontier call, seven progress events,3.64478324 main CPU seconds, actual20/20CPU and2GiB limits. No model forward, geometry, mass or RNG calls occurred.

| Retained-value oracle | Remaining nominal FLOPs | Fraction of original |
|---|---:|---:|
| Original unpruned panel | 8,375,735,504 | 100% |
| Exact error ordering, original fixed threshold | 2,180,664,002 | 26.035% |
| Exact error ordering, self-consistent threshold | 1,597,294,956 | 19.071% |
| Exact error/cost ordering, self-consistent threshold | 1,471,873,760 | 17.573% |
| Fractional optimistic cost lower bound | 1,470,771,585.224 | 17.560% |

Thus the feasible selection and fractional relaxation bracket the best integer subset's remaining cost between1.470772B and1.471874B nominal FLOPs under the declared retained-value model. The bracket width is0.0749% of the feasible remaining cost. This does not require proving the displayed integer selection is exactly optimal.

The ratio selection's removed numerical error is approximately5.8484693e-5; its conditional one-SE limit is approximately5.8568639e-5. Exact squared-bias/nominal-variance is0.99713544. These concern stored numerical errors and original stored p, not true integration errors or an independently established estimator MSE. The existing floating report has the same displayed fixed-threshold selected count/cost here; its missing row identities and floating/tie behavior were not reconstructed.

29,031 rows have zero mass cost and none has positive stored error. Ratio selection omits zero-cost rows and selects19,852 positive-cost rows; error-prefix counts include zero-cost rows, so selected row counts are not directly comparable. Fixed outside-mass cost remains205,974,641 Ops; mass-call costs include internal triangulation geometry. The exact sum errors differs from stored error_total by the recorded small floating discrepancy; this is preserved, not corrected.

## Interpretation and limits

[measured, independently audited] Cost-aware ordering plus the self-consistent threshold improves substantially on the former error-first fixed-threshold heuristic on this fixed panel. The earlier heuristic is not an upper bound on possible pruning savings.

[inferred, conditional on the reviewed model] The fractional relaxation narrowly brackets the optimal integer subset for these retained numerical values. It excludes stronger savings only within the stated fixed-cost, whole-row removal, one-SE retained-value model. It does not exclude different representations, earlier geometric pruning, better certificates, other estimators or other models.

The oracle needs all original numerical errors and costs. Neither that advice nor selection work is free deployment work. There are no certified per-row mass intervals, new checkpoint-byte reconstruction, historical native/runtime binding, new true-accuracy reference or H16T10/noise extrapolation. Model identity is the original artifact label and collector provenance. This saved calculation is not a new estimator leaderboard result; strict/lenient MSE bars remain untested.

Full exact inputs, sorted identities, all prefix/breakpoint records and eighty boundary decisions are in target_frontier.json.gz. Summary/provenance are target_results.json and target_gate.json. Independent saved audit4b8e6bc4/sharede8873c78 verifies all51,552 rows,22,521 positive costs,22,522 breakpoints,all methods and80 decisions without helper/model/mass replay;2.773304452reviewCPU. No further target call is planned.
