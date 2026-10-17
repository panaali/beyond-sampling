# T134 bounded full-suite cost recount

Claim b122b61 preceded protocol0b80351 and implementation2906de4. All471 fixed T99 models/noise copies are reconstructed exactly (weight SHA checks), with saved truths and original estimates retained. No new model, noise, truth, geometry, recipe or threshold was selected. T128 source/cost mapping is pinned and independently reviewed.

All471 cases preserve the exact float-hex estimate, every family result, original source-era counter, trace count, constraints and segments. Current helper diagnostic X charges are removed only when reconstructing the old author counter, with AST and runtime evidence retained. Scalar sampler costs are112/160/216 atH4/H5/H6, not the historical72/105/144 proxies.

| H, T=3 | Cases | Largest individual squared-bias ratio | Largest group ratio | Mean expression count | Native/original sample-equivalent factor |
|---|---:|---:|---:|---:|---:|
|4|157|0.1923764203 (seed1)|0.03980899970 (clean)|606923.809|1.87249–1.91703|
|5|157|0.003886719292 (seed0)|0.003417828251 (clean)|832982.892|1.91041–1.96040|
|6|157|8.850394381e-9 (abs0.05_n1_40)|1.091319704e-10 (abs0.05_n1)|1131076.529|1.93913–1.97855|

Each individual ratio is bias² divided by p(1−p)/N, N=expression count/scalar sampler count. Each group ratio is equal-weight mean squared bias divided by equal-weight mean of those per-model sampling variances. It is neither mean individual ratios nor a single common-budget comparison. All471 individual and21 group ratios are below1 using the saved approximate truths.

Scope matters: each H contains five clean seeds but noisy copies of only one historical anchor (H4seed0/H5seed1/H6seed3),24 relative perturbations plus128 absolute0.05 copies. This does not establish the full ladder protocol across all seeds' noisy copies. Near-zero H6 discrepancies are numerically indistinguishable from exact agreement at some points; no interval precision or extrapolated population guarantee follows. H16T10 remains unsolved.

Named counts include written scalar/array/integer expressions and special calls under the independently reviewed T128 convention. Implicit iteration, memory/control, underlying library/hardware details and universal precomputation remain excluded. This is an abstract expression comparison, not complete implementation or hardware cost. Per-model model-dependent setup is included and module state is reset each call. Source instrumentation itself is excluded from estimator arithmetic.

All24 bounded runs took372.8883CPU seconds in total including diagnostic import/reconstruction/instrumentation, split112.7477/121.2396/138.9009 byH. Preflight added2.505CPU; these timings are not solver-performance comparisons. Every chunk was committed separately. No chunk failed parity or exhausted its frozen cap. An early runner exit-status repair is preserved in a14a7bb; no failed numerical result was silently discarded.

Reproduce preflight with `python -m m16.approaches.fibre_suite_cost.preflight`; each original measurement used `python -m m16.approaches.fibre_suite_cost.run_chunk --H H --start S --count 20`, S=0,20,...140, with one thread, memory cap and primary lock. Existing files are deliberately protected against overwrite. Recompute aggregate only with `python -m m16.approaches.fibre_suite_cost.summarize`; it asserts complete unique coverage of all471 indices. Independent T55 source/denominator review supplied; final raw aggregate review requested against64bdf33.

This closes the frozen recount, not a method family. Remaining alternatives/checks: formal uncertainty of saved truth, full noisy panels at every clean seed, implementation-level cost conventions, generic H7 checkpoint provenance, T4 primitive/gate repairs. Shared source and DRI-owned ladders remain untouched. Root/DRI should register the result; only DRI may confirm a claim.

Independent final T55 review ea53df3 verifies all471 raw identities,942 site ledgers,25 source hashes and21 exact-rational group denominators without rerunning estimators. All stated finite-panel conclusions pass; numerical-truth and incomplete-implementation-cost qualifications remain. DRI confirmation is still separate.
