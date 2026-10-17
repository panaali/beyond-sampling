# T130 bounded positive mass cost study handoff

Claimd3b45e9 preceded implementation. Same6347 captured quadratic rays; no source/geometry/model changes. Cheap separating-center radial bounds, explicit24-term determinant, interval shortcut and fixedGL2..32 rules, two predeclared tolerances. All setup/failed selection is retained; saved positive references are used only for evaluation.

Analytic controls54 total:15 return and pass their bounds+5e-14 floating allowance;39 hit the fixed finite order limit. Exact16-case center/determinant checks preserve up to1.78e-11 determinant relative error; degenerate/nonseparating inputs are explicitly unresolved. T55 independently confirms real-arithmetic formulas and named determinant/kernel/Duffy-plan counts. This does not establish floating interval containment.

| Objective | Returned | Interval/rule/unresolved | Covered mass error | Bound sum | Named operations | CPU |
|---|---:|---|---:|---:|---:|---:|
|1e-7 per cone|6347|3204/3143/0|+8.8647292e-6|1.1969158e-4|89,513,307|4.728s|
|1e-7/6347 per cone|6315|854/5461/32|+5.2770493e-10|1.5989900e-8|364,952,602|6.223s|

First objective accuracy .47700426126492024 differs−8.8647292e-6 from reference; second has no full aggregate estimate because32 cases remain unreturned. No returned-value bound violation against saved positive references plus their individual conditional bounds and floating allowance. Passing individual1e-7 masses therefore does not control network accuracy at1e-7.

Named counts include executed scalar setup/selection, comparisons/sqrt/absolute operations,39n³+1 integrand arithmetic and18n³+3n per cold Duffy plan. T55 identified ordinaryπ-constant expressions outside the scalar wrapper; additional25,266/34,602 operations have been charged from full call ledgers. Original initial ledgers are preserved; numerical values were not rerun or rewritten. Legendre-rule construction is timed but its portable arithmetic remains unaudited. Counters use a stated convention, not a complete portable-cost certificate. Observed CPU and named work are much larger than the previously measured source primitive; no exact portable cost-ratio claim follows.

This finite positive tensor-rule variant fails the cheap-reference replacement objective on the captured corpus, and its finite cap limits coverage at the stricter budget. It is not a failure of positive representations in general. A plausible analytic rescue is a Dirichlet first/second-moment interval for q^-2, potentially avoiding most tensor nodes; stronger center/KKT bounds, stable determinants and distinct low-dimensional integral representations remain open. Those need separate frozen scope and independent verification, not a larger blind order sweep. T55 raw numerical/cost-ledger check pending; DRI retains classification authority.
