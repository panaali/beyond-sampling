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
geometry. Eight fixed12-parent primary-lock batches each <=110CPU,2GB; save complete failure
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
Bounded full trace complete; independent raw aggregate review add47fb passes.

## Results
All eight geometry batches reproduce all35098 ordered ray arrays,8455
terminal cells, and the35650854 legacy geometry count. No trained Gaussian
mass was recomputed. Independent source review by56 and T55 passed before
the trained trace; T55 independently derives the192-per-node and512-sampler
counts.

| Category | Explicit named units |
| --- | ---: |
| Full geometry, correcting seven extra diagnostic return subtractions | 42,396,504 |
| Mass from existing fixed branch census | 510,972,352 |
| Cold chamber construction | 30,288 |
| Executed original plus candidate rule setup expressions | 1,624 |
| Visible cold total | **553,400,768** |

Opaque additional kernels:2,659,906 4x4 inverses (including48 cold chamber
inverses),35,098 determinants,275,897 matrix-rank/SVD calls, and four
model-independent leggauss generations. Rank shapes range1x4 through24x4;
208,370 calls use2x4. Membership/hash/tuple search, short-circuit machinery,
memory/indexing and native implementation internals remain unpriced.

Conditioned on the saved single-checkpoint reference and its integration
bound only, equal-bias sampling break-even total cost is about609.4 billion
of the declared units. The visible subtotal's squared-bias comparison is
about0.000908, leaving large algebraic headroom for opaque work. This is
**not a complete cost comparison or a sampling win**: these unknowns and
floating geometry/reference uncertainty remain. Other seeds/noisy copies
are outside this audit. T150 independently studies inverse dispatch costs.

Trace functions used50.542638CPU total, maxRSS202000KiB. Timings include
model loading and repeated source/rule/chamber setup inside each batch, but
exclude initial module imports and final JSON serialization. They are review
costs, not production timing. Cold setup is composed once for one estimator;
the eight review launches are retained as consumed diagnostic compute.

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

## Primitive phase
Original missing single-iterable min/max tracing raised explicitly during
controls; no trained geometry ran. Failure/source preserved da7689d, then
streaming counting was added without materializing or changing iteration.
Three mass branches preserve exact baseline outputs and legacy counters.
Inclusive explicit expressions: wide24=4808, thin96=18632, cutoff=184, plus
opaque inverse/determinant calls. All fixed-array function shapes remain
recorded. Same-wrapper dense scalar sampling is exactly512 units for H8T4.
Primitive CPU.099790s; source review required before full geometry trace.

The geometry control found a metadata-only failure when a stack's input list
contained differently shaped arrays. c04bc38 preserves the failure before
trained execution. Recursive shape descriptors repair metadata without
changing any numerical operands. Full96-parent zero-model baseline and
traced geometry now match all72 ordered ray callbacks,96 cells, and17472
legacy operations (.206571CPU). Trained tracing will use the source's
existing eight12-parent partition to keep each batch bounded; repeated
setup is measured separately and composed once for one cold estimator.

## Existing branch census
No trained integration was rerun. All35098 stored indices/flags appear once:
wide10245, thin24780, cutoff73. Scaling the branch controls reconstructs the
legacy404438808 exactly and yields510972352 inclusive explicit units plus
2659858 opaque4x4 inverse calls and35098 opaque determinant calls. This
ledger is deliberately separate from the later independent T150 inverse
kernel audit and T144 determinant replacement work.

The explicit-expression convention counts each special-function call or
power as one named unit and reductions by their declared algebraic count.
Python/NumPy's internal reduction/power implementation, vectorization, and
branch machinery are not thereby certified; these are additional native
implementation qualifications, alongside LAPACK and memory. Source diagnostic
Ops/STATS accounting is excluded, while actual integer decisions are retained.

Independent owner56 read-only review of1ed0d55 found no numerical blocker
to the eight trained batches: ordered-ray callbacks, native dense/reduction
counts and original source semantics are preserved. Specific missing
control categories remain explicit in `unpriced_control_sites.json`:
membership/identity comparisons, set/hash/tuple searches, Boolean short
circuit machinery, implicit loops and dispatch. These are not free in any
whole-implementation claim. The reviewed inventory is a common abstract
expression convention with opaque categories, not a portable total.

## Reproduce
Use the project interpreter, one BLAS thread,2GB cap and the appropriate
shared lock.
```sh
python m16/approaches/plackett_cost_review/controls.py
python m16/approaches/plackett_cost_review/geometry.py controls
python m16/approaches/plackett_cost_review/mass_summary.py
# Run geometry.py with each integer0..7, one bounded primary-lock phase each.
python m16/approaches/plackett_cost_review/summarize.py
```
Every source, model, archived reference and trace source is pinned in
`inventory.py` / `trace_provenance.json`. Scope remains an explicit-expression
ledger with opaque categories; it is not a full hardware instruction count.

T55 add47fb independently recomputes every site ledger, the seven extra-return
correction, single cold setup, opaque counts and553400768 visible total.
The conditional break-even algebra is independently reproduced, retaining
all unpriced categories and no full-cost/sampling-win claim.
