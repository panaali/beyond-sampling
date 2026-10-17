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
