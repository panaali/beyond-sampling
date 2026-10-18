# T267 bounded numerical handoff

Both requested small rungs are complete and independently checked. The method
cancels identical oriented internal facets, then cones the remaining boundary to
fixed orthant anchors. It reduces integration terms, but its measured construction
and recording cost exceeds the integration saving in both runs. This is a scoped
implementation outcome, not a negative result for every boundary method.

| Rung | Raw producer | Independent saved audit | Terms before → after |
|---|---|---|---:|
| H4T4 | `0059948f`, shared `ea1c1c70` | `7fe3ac0b`, shared `a63d9360` | 5,058 → 3,056 |
| H4T3 | `b1337926`, shared `6d7f45c1` | `e18bedc1`, timing note `9b2288f7`, shared `497c1d14` | 244 → 200 |

H4T4 source gate `593cc500` and immutable execution gate `9d9d9937` were published
before execution. H4T3 source gate `8c0b1de9` and immutable execution gate
`29acef0e` (shared `887a06bc`) likewise preceded its sole run. The independent
H4T3 checker was source-gated by `aba74728`, shared `a7de7a08`. No failed producer
attempts or completed scientific reruns occurred.

H4T4 uses the existing archived exact complex and fixed GL20 integration. It saves
39.58% of fixed-rule nodes; construction/recording takes 2.518 CPU seconds versus
0.469 CPU seconds saved integration. Its signed absolute determinant weight rises
42.02%, worsening the real-rule truncation upper. H4T3 explicitly charges one new
exact capture from the existing weight archive, then uses closed solid angles.
It saves 18.03% of calls; chain construction/recording takes 0.0370 CPU seconds
versus 0.00377 seconds saved integration. New discovery components take 0.6004 CPU
seconds. See `COMPARISON.md` and the rung reports for the full recorded counters.

Numerical total differences are 3.33e-16 and 1.11e-16, respectively. These are
consistency diagnostics, not floating-point accuracy certificates. Independent
checks cover the retained exact chain identities, incidences, readout/rank
constraints where scoped, and saved scalar arithmetic. Original global geometric
coverage/disjointness, historical checkpoint identity, library-function accuracy
and full native cost retain their explicit qualifications.

Different anchors, compatible subdivision merging, cross-chart matching,
amortization and direct lower-dimensional boundary integrals remain untested.
Neither finite result supports a general impossibility claim, large-width scaling
law or final ARC/HP1 estimator. T278 owns the separate C1 numerical discriminator.

Please register the combined bounded result in INDEX and close T267's requested
H4T3/H4T4 phase. Any alternative boundary experiment requires a fresh scoped row;
no outcome-driven extension or completed replay is in progress.
