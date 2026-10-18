# T256 exact affine ranges on gap-constrained boxes

## Question
Can we sharpen a local network enclosure by optimizing its affine input term on a labeled top-gap domain without enumerating all polytope vertices?

## Method
PROOF.md reduces the bounded linear problem to the second coordinate and finitely many clipping breakpoints. PLAN.md separates written proof, exact controls, inherited-model integration and probability evaluation. The core coordinates are physicalx; inheritedv=2x must transform BOTH endpoints/objective or the gap explicitly.

## Status and results
Written prerequisite only, claimc02f8a1d, independent written reviewer34. No implementation or numerical result. T240's intended physical-gap campaign remains failed; its independent actual-gap diagnostic b381f518 is retained, with physicalgap1 untested.

## Compute accounting
Source/document work only. Zero code/LP/vertex/network/CDF/checkpoint/array/RNG calls. O(T^2) is an exact-expression bound for one affine objective and one ordered pair, not a complete FLOP price; all-pairs and rational bit costs remain separate.

## Claims
[inferred, pending independent review] Conditional endpoint completion and all clipping breakpoints give exact affine minima/maxima.
[speculative] Restricting affine ranges could increase final network certification. Dependent remainders, mass integration and runtime can remain limiting.

## Alternatives not ruled out
T252 partial-box mass integration, other domain partitions, stronger dependent-error representations, exact suffix laws and retained old certificates. No negative result or learned-model conclusion yet.

## Failure modes
Missing the physical-coordinate conversion, omitting a kink, equating a touching domain with positive mass, ignoring g0 rank ties, or assuming final ReLU bounds improve monotonically after slope changes would invalidate later claims. The proof addresses only a single affine range; all subsequent steps require their own gates.
