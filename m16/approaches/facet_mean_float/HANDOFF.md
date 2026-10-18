# T175 bounded numerical evaluator handoff

Baseline20 dd6d3129 independently4895c343 preserves narrow40 relative failure; entire saved float summation was exact, so compensation alone cannot fix term-evaluation error. Candidate-only heuristic rescueafac123f independentlydd19f41b passes20 with2MP80 calls/24decodes and18 unchanged baseline results.

Frozen544 dbdfcb7d after source47e4faff/910 gate passes544cell and16pooled gates,155MP80 calls/1860decodes; independent saved review8f922a46 passes all decisions/errors/pooled arithmetic without replay. No new exact preparation, angular-reference, model, MC, covariance or downstream calls. Exact prep is model-dependent real work and all original counts remain. The heuristic is not a rigorous error certificate.

Direct floating retains31 cell-relative failures, but retrospective direct pooled16 means already pass the same tolerance (max2.169e-14 versus hybrid1.873e-15). Fallback necessity is therefore not demonstrated for this model’s pooled tolerance. Total .151824240CPU is descriptive, not a matched speed or complete-cost result.

All scalar failures and every saved attempt remain immutable. Root/Claude registry requested with independent8f922a46. Next DRI priority is T188 same-future merging;34 coordinates T187 and owns pair-moment work, no expanded mean/covariance computation here.
