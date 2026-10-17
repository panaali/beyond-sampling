# T146: independent cost inventory for the repaired T4 calculation

## Question
Does the improved fixed T140 quadrature remain plausible at the same compute
as sampling once the actual dense work and unknown kernel costs are exposed?

## Method
Existing Ready T146 claimed and pushed f6487ba before implementation. Pin
T131 estT_snapshot, H8T4seed0 weights, error capture, and T140 candidate/count
summary. Do not rerun35098 Gaussian masses. Two bounded parts:

1. Source inventory and exact native-expression trace of the mass routine
on three synthetic branches: wide identity, thin block-correlation.99, and
source determinant-cutoff diagonal1e-13. Compare baseline/instrumented
outputs and legacy counts exactly. Scale the respective fixed-shape ledgers
by the existing archive's wide/thin/cutoff counts. Rebuild branch counts from
stored batch flags, not a new matrix-inverse replay. Source array functions,
scalar arithmetic, comparisons, dense products, reductions, special functions
and all linalg calls receive an explicit category. Numpy inverse/determinant/
rank/eigen kernels remain unknown implementation-dependent work, with call
shapes/counts; they are never silently replaced by n^3 legacy charges.

2. Trace full cold exact geometry with orthant replaced by a zero-returning
mass recorder (no added fake operations). Source geometry must reproduce
8455 terminal cells,35098 ordered saved ray arrays and historical geometry
counter35650854. Trace halves construction separately because source uses
its own discarded Ops objects. A native-expression AST wrapper delegates
every unchanged numerical operation, preserving numerical order. Source,
model and mass archive are pinned; instrumentation controls precede full
geometry. One primary-lock bounded run <=110CPU,2GB; save complete failure
and do not interpret an incomplete ledger as a cost result.

Reuse the independently reviewed T121 AST approach read-only, extending
local wrappers for this source's NumPy functions. Arithmetic of explicit
integer indices/control expressions is counted alongside floating work.
Model-independent GL and chamber setup is disclosed cold and separately;
no cross-model learned cache is free. Same-AST sampling computes H8T4 with
first known-zero hidden update skipped, dense mul/add distinct, ReLU and
first-index prediction plus correctness comparisons counted. Expected512
units is a derived control, not an assumed denominator. Input generation
and target labeling follow the existing free-input comparator convention.

All memory allocations/copies/indexing, implicit interpreter/control loops,
BLAS/LAPACK internals and data-independent rule generation remain named
exclusions or opaque counts until specifically resolved. A partial numeric
sum is not a complete portable or hardware cost. Report nominal headroom
against this one checkpoint's squared bias only, retaining reference
uncertainty and no noisy-suite/rung success claim.

## Alternatives and falsifiers
1. Corrected dense work dominates legacy counters: source trace quantifies it.
2. Correcting both sampler and estimator conventions cancels some inflation:
   common native-expression controls distinguish this.
3. Instrumentation changes geometry or source branching: ordered ray parity
   and legacy counters must fail before cost conclusions are drawn.
4. Unknown rank/inverse iterations prevent a full portable bound: retain exact
   call shapes and state the limitation instead of inventing kernel costs.

## Status
Protocol frozen; pinned source inventory and primitive controls next.

## Results
Pending. No new mass, accuracy, seed/noise or recipe experiment.

## Compute accounting
One thread,2GB; review lock for <=20CPU controls, primary lock <=110CPU for
full geometry. Count imports/setup/failed diagnostics separately. Source-era
legacy counters are preserved as evidence, not presumed complete FLOPs.

## Claims
[inferred] Dense products execute their full dimensions despite zero entries.
[speculative] Corrected source cost still leaves room for sampling comparison.
No complete implementation count or challenge win is claimed from omissions.

## Alternatives not ruled out
Counted fixed-dimensional inverse/rank replacements, genuinely sparse source,
positive mass primitives, alternate fixed quadrature maps, and full noisy-suite
accuracy/cost validation. T145 separately measures recipe pruning bias.

## Failure modes
Double-counting legacy and new counters, granting unimplemented sparsity,
ignoring integer decisions on one side, hiding LAPACK work, changed tracing
semantics, setup exclusions, and reading a single-checkpoint bias as MSE success.
