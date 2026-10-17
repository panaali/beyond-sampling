# T110 trained T4 class-region partition diagnostic

## Question
Does the quadratic ordered-pair or cubic nearest-facet initial partition preserve exact trained H4T4 accuracy while reducing downstream work, and how does the frozen approximate recipe change? Claim6e0392f was pushed before implementation. This follows the independently reviewed T3 partition phase; it does not duplicate Claude's gate certificates or cone-mass implementations.

## Method and frozen protocol
Read-only imports from `class_region_families` pin families.py SHAe1c93ac9096bcb37f9f9f39bbb79b1c2b1d65314000a3029dc34919768ac6a6a, tie_repair.py SHA11eaa7354fb864a3d880727dcbf8b53fa5fa4cc22c5c0c7fd3138d78f81a4de8, estT_snapshot.py SHA0a2db2eb14e37e0d6575630c15b145d41cdedba5e977e603662576fd3700c5cb. Activate exact-zero output tie repair, retain every other tolerance and mass rule. Existing H4T4 seed0 checkpoint SHA83ddefeaee737c73613cdc786b01b3bb79c0007bd6c1e42a454302c4801dd146 only. Independent previously validated reference.477013125994119 is diagnostic, never an estimator input.

Freeze six cases: original/quadratic/nearest partitions × exact/recipe, eps.1. Order exact original/quadratic/nearest then recipe original/quadratic/nearest. Before trained runs, validate initial counts96/24/72, total/class masses and analytic constant-label first-index controls. Each trained run gets separate50CPU/200M named-operation cap; cooperative checks in operation additions and hard55CPU limit. Save completed-family error sums/counts plus partial cost when stopped, never report a partial sum as full accuracy. No noisy copies, node/threshold sweep or higher rung.

## Falsifiers and alternatives
1. Fewer initial families may leave equal or more downstream gate pieces/simplices. Report initial counts, gate splits, terminal cells, mass simplices, setup and totalCPU and named costs separately.
2. Approximate reference geometry may change even where exact integration is invariant. Report every frozen recipe value, signed reference error and difference from original; no winner-selected variant.
3. An exact discrepancy can arise from inherited finite mass quadrature or floating cone topology. A discrepancy above5e-7 triggers a separate predeclared primitive/topology diagnostic; it does not refute the real-arithmetic partition identity. Existing old T4 mass primitive is not a certificate, nor is agreement a proof against every thin-cone error.
4. Constant output may expose tie ownership independently of trained accuracy. Use a cheap allzero H1T4 model, whose exact answer1/4 follows exchangeability, on all partitions with tie repair. No full trained zero-readout repeat.

## Status
Protocol frozen before implementation; numerical phase pending.

## Results
Pending. No accuracy/cost or sampling advantage claimed yet.

## Compute accounting
One thread,2GB, standard primary lock; <=110CPU per cycle. Each run imports/constructs its own rules and geometry; full phase CPU includes setup, with import CPU separately recorded. Old historical initial construction proxy was missing: report its measuredCPU and operation count as unknown, not zero. Named operation totals inherit unaudited NumPy/rank/special-function proxies and omit some Python overhead; no portable cost or total-polynomial assertion. Checkpoint I/O and diagnostic references are separated from production work. No random model inputs or Monte Carlo truth.

## Claims
[inferred, prior independently reviewed] The initial-family partition is mathematically exact; initial counts do not imply downstream polynomial complexity.
[speculative] A coarser initial partition reduces useful end-to-end work at T4.

## Alternatives not ruled out
Other chart choices, exact boundary representation, all-facet gate certificates, better thin-cone mass primitives and adaptive geometry. No method family is dead.

## Failure modes
Inherited clipping tolerance, nearly singular cone mass, exact-zero versus near-zero ownership, uncounted numerical kernels, truncated runs and cancellation of integration errors. Preserve all caps and failures explicitly.

## Preserved initial control failure and frozen discriminator
The original partition H1T4 zero-readout control gives .249999976701354, a−2.3299e-8 discrepancy from.25, exceeding the frozen1e-8 test. It completes96 families/240 terminal cells/216 mass simplices; initial family mass normalization itself passed. The full failure ledger is retained in `initial_control_failure.json`; no trained case has run. This small discrepancy is consistent with the inherited Plackett quadrature, but that explanation must be tested.

Freeze a controls-only comparison for all three partitions with the unchanged default12/48 graded rules, doubled24/96 and quadrupled48/192 rules, preserving identical gates/geometry and analytic target. Report every result, error, cost and CPU. If tighter rules restore the analytic value and leave all geometry counts identical, classify the original control as a finite mass-rule accuracy failure rather than a partition/tie failure. Do not silently replace the default trained recipe or erase its failure; full comparisons retain default quadrature and explicit error limitations. Separate stronger mass rules or an independent mass method would require a new scope.

The frozen mass diagnosis completes3.15CPU. Default constant errors are−2.33e-8(original),−7.37e-11(quadratic),−1.96e-8(nearest); doubling rules reduces all to<=1.46e-14 and quadrupling to<=8e-15, with identical geometry counts. Per-family network error versus directly integrated initial mass has default residual<=1.64e-9, falling<=2.83e-15 after doubling. Summing initial families by ordered top-pair/sign and comparing independent exact rank-sign probabilities gives default chart error<=1.53e-9, dropping<=2.36e-16. This discriminates finite mass-rule error from the tested partition/tie calculation. It is finite evidence, not a universal floating geometry proof.

The original1e-8 control failure remains preserved. Proceed with the frozen trained comparisons using unchanged default mass rules, explicitly as numerical/cost diagnostics; no exact-value certificate is inferred. The independent trained reference comparison retains its predeclared5e-7 discrepancy trigger. Any larger discrepancy gets a separately frozen follow-up before an invariance conclusion.
