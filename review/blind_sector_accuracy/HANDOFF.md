# T268 bounded numerical handoff

The saved H4T4 sign-sector correctness baseline is complete, independently audited and published. The richer conditional predicted-class objective remains open.

## Result

All 5,058 archived correct simplices and their paired reference intervals were assigned to the sixteen exact sign orthants. Every sector has strictly interior conditional correctness bounds. For the five blind sectors (all negative or exactly one positive input), total Gaussian mass is 5/16 and conditional accuracy is approximately [0.567741736598, 0.567741800692]. The best deterministic sector-only binary predictor of correctness has disagreement mass approximately [0.101599821143, 0.101599841172] over the full input distribution. Exact rational endpoints and all sector results remain in `output/results.json`; displayed decimals are approximations.

This is not an obstruction to a predicted-class quotient: the true second-rank label varies within each sector, so even a constant output can have variable correctness. Nor is binary prediction loss an accuracy-estimation MSE lower bound. The result does not reject compact conditional probabilities, richer summaries or cheaper estimators.

## Evidence and accounting

- Numerical source `f875b6eb`, pre-run durability repair `f9379258`, qualification `26c93a5e`.
- Independent source/theorem/resource gate `a4c7046c`, published `7c7bf23e`.
- Execution scope `69a88253`; immutable gate `7aca5771`, published `4c606d21` before the sole separate launch.
- Raw `e893ef82`, published `b9c1f28f`: seventeen controls, 5,058 rows, twenty chunks, sixteen sectors and 76 events; 3.848091687 internal CPU seconds, 78,484 KiB peak RSS, no failure or retry.
- Independent checker source `d9d3004f`, source gate `4605aada` / shared `56e790e3`; independent final saved audit `27db195b` / shared `7026e40e`: exact Gaussian elimination and midpoint-distance loss, all 46 pins and complete coverage, 0.771083525 CPU seconds, 79,756 KiB peak RSS, no author calls or failures.

The inherited reference reports 38.605260022 seconds of composed successful cold construction/enclosure cost. This is reused advice, not free estimator setup. The new contraction and review costs are separate. Selected arithmetic counters do not establish native FLOPs. Geometry coverage/disjointness, original numerical enclosures and platform premises remain inherited. No model, new geometry, Gaussian integration, RNG or fit was run in this phase; no target-scale, noisy-panel, FPE or held-out scoring claim follows.

## Release

The tested numerical baseline can be registered by the DRI. Original T268 class-law construction remains unresolved and requires a concrete separately scoped test rather than extra outcome-selected sectors. All sixteen sectors and the predeclared five-sector objective were retained; fixed-output baselines, class distributions, richer summaries and new models were not tested. No family is marked dead.
