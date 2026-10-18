# Fixed H4T3/H4T4 comparison

This tests identical-facet cancellation followed by coning to one fixed anchor
per orthant. It is not an exhaustive test of boundary integration methods.

| Quantity | H4T3 | H4T4 |
|---|---:|---:|
| Original integration terms | 244 | 5,058 |
| Reduced signed terms | 200 | 3,056 |
| Reduction | 18.03% | 39.58% |
| Cancelled canonical facet groups | 236 | 7,697 |
| Remaining zero-anchor cones | 60 | 1,782 |
| Absolute determinant-weight increase | 1.417% | 42.019% |
| Integration rule | Closed trihedral angle | Fixed GL20, 8,000 nodes/term |
| Baseline integration/recording CPU | 0.01107 s | 1.23368 s |
| Reduced integration/recording CPU | 0.00729 s | 0.76451 s |
| Chain construction/recording CPU | 0.03704 s | 2.51836 s |
| Absolute difference of numerical totals | 1.11e-16 | 3.33e-16 |

The geometry is different across rungs; these timings are individual instrumented
runs, not a dimension-scaling fit or an uninstrumented throughput benchmark.
H4T3 charges a new exact capture (0.60039 CPU seconds for the recorded discovery
components). H4T4 reuses an archived complex, retaining its inherited discovery
cost separately. Neither comparison treats setup or recording as free.

For both rungs, the measured chain construction/recording cost exceeds the saved
integration time. Fewer terms therefore do not demonstrate a cold end-to-end
speedup in this implementation. Reuse across repeated integrands could change
the amortization, but no such workload was run. Other anchors, compatible face
subdivision merging, cross-chart matching and direct lower-dimensional boundary
formulas remain untested.

H4T4's real-rule truncation upper worsens from 2.966e-8 to 4.212e-8 because signed
absolute weight grows. H4T3 uses no quadrature, but its floating atan2 evaluation
still lacks a directed error certificate. Small differences between the two
computed totals are consistency diagnostics, not proof of their true accuracy.

H4T4 is independently saved-audited (`7fe3ac0b`); H4T3 raw `b1337926` passes
its separately assigned audit (`e18bedc1`). Global geometry completeness/disjointness,
historical checkpoint identity and full native cost retain the qualifications in
the respective reports. No ARC/HP1 estimator, noisy-model or large-width claim is
made.
