# T255 bounded evidence review

Frozen owner head: `2aad2d28b49a6c336b8ff60b9c2e42964f7d4948`.
Source package `b8d5cb1b` independently reviewed by910 (`22330943`, shared `32585752`). Checker source `7d549811` independently cleared by56 (`ea7b5544`, shared `23d08f2c`). Immutable execution gate `acdbd1c3` was published at `7f687911` before the sole saved run `9caaad75` (shared `2f9358e8`). Final separate saved arithmetic review by910 is pending under delegation `74c79d2f`.

The fixed300 records reconcile to300,000,000,000 inputs and285,935,243,665 correct outputs. Exact saved accuracy is57187048733/60000000000. Nominal binomial SE is3.85938468025e-7; the exact between-block mean SE is3.41834782746e-7. These reproduce the displayed arithmetic. They do not authenticate historical model/runtime/stream independence or establish an exact accuracy bound. The block0 float32 comparison is+29 net correct outputs; the individual prediction discordance is unknown.

For k6,245/3811 completed subtrees retain1,473,541 cells and1,459,727 reported histories. The recorded cell extrapolation is5615664751/245, approximately22.921 million and30.619209 times the k5 cell count. The recorded numerical retained mass is about0.088291982. Deadline-selected completions are not generally an equal-probability sample: random task dispatch does not remove completion-time selection. Our independently reviewed finite counterexample invalidates the generic unbiasedness argument, without measuring actual bias magnitude or direction. `sec_cpu` sums worker wall durations and is not a process-CPU ledger. Partial-mass normalization, history/cell distinction and finite numerical integration remain qualified.

No author model, MC, RNG, geometry, array or mass-integration call was replayed. The saved arithmetic run records6 events,0.008875512 CPU seconds before final save, actual20/20-second CPU and2GiB/2GiB address-space limits. Independent exact target count evidence remains T219 k<=4; this review does not extend it to k5/k6.

Plausible alternatives remain open: actual selection bias could be small, complete per-subtree records could support a different estimator, and additional runtime/model/stream provenance could strengthen MC interpretation. No scientific adoption, family impossibility, full-cost or architectural accuracy-ceiling claim follows. Claude retains registry and scientific acceptance.
