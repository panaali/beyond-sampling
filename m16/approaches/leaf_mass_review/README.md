# T131: independent leaf-mass overcount diagnosis

## Question
Does the H8T4 seed0 exact-mode total-leaf mass excess come from clipping/triangulation, finite Gaussian quadrature, or both? Final3 reports1.000888 total leaf mass and attributes the defect to topology. Eight random cone Monte Carlo checks do not exclude systematic quadrature error on many trained thin cones; prior T119 already found finite mass-rule errors on H4T4.

## Method and frozen first discriminator
TASKS-only claim49947ed precedes implementation. Pin latest origin estT.py (symbolic incidence), m43 dependency, H8T4 seed0 local checkpoint and final3 evidence. Copy a read-only engine snapshot into this owned folder; never edit shared estT. No new MC, noise or full-rung evaluation.

First isolate geometry from Gaussian mass. Run exact neural gate splitting with a terminal callback that captures every terminal leaf's existing pulling simplices; replace only Gaussian orthant integration by a recording stub. For each of96 initial half-chambers, choose the fixed interior dual anchor equal to the sum of its inward facet normals. It is positive on every parent ray. Normalize every parent/leaf simplex ray to anchor dot ray=1. Sum absolute determinants of normalized rays: a correct disjoint partition must conserve this affine section volume, independently of Gaussian density. Record parent/leaf volume ratio, all node counts, and inequality violation residuals. Capture all resulting rays and their parent/leaf identities for independent same-geometry positive mass checks; no fitted selection.

Predeclared controls: orthant and one coordinate-difference cut, then record exact dyadic Fraction determinant checks for the first two parent partitions and the parent with largest floating volume residual (ties first index). Exact dyadic rays reflect captured floating geometry, not exact-real neural coefficients. Fraction volume comparison may reveal tiny arithmetic mismatches; never call approximate conservation a universal topology certificate. If a partial run reaches55CPU or200M inherited named-operation cap, save completed parents and unresolved remainder. One thread,2GB, primary compute lock.

Next scope requires freezing before implementation: replay same captured leaf simplices with independent positive Gaussian quadrature/radial bounds (T119/T123/T126), starting fixed first parents and largest geometry residual. This separates mass-rule error from geometric overcount; a numerical reference discrepancy alone cannot identify topology. Preserve source mass and every source/fallback failure. No shared repair until a concrete defect and independent review exist.

Alternative1: finite Plackett integration error accumulates on thin cells (same-geometry positive replay). Alternative2: symbolic clipping/triangulation overlaps cells (density-free section conservation, exact dyadic checks). Alternative3: terminal/output tie handling or checkpoint/source mismatch (hashes and separate gate/output ownership). These remain distinct until tested.

## Status
Protocol frozen; capture pending.

## Results
Pending bounded discriminator.

## Compute accounting
Charge checkpoint loading, engine/rule setup, all splitting/triangulation, normalization/determinants, capture/checkpoints and Fraction verification. Named-operation counters are inherited partial proxies, not a portable total. No model sampling or original accuracy benchmark is performed in this phase. Capture is new work and cannot be treated as a free reference for later methods.

## Claims
[inferred] Probability sums above one establish an inconsistency, but do not by themselves distinguish geometry from numerical integration. [inferred] Section-volume conservation is a necessary independent topology check, not sufficient to rule out compensating overlap/omission.

## Alternatives not ruled out
Improved mass rules, exact clipping, robust adjacency/rank predicates, output-tie corrections, adaptive chart selection and new representations. No whole-family negative or solved-rung claim.

## Failure modes
Floating nearly singular rays, cancellation in determinants, tolerance-based geometry, inherited incidence errors, inaccurate mass primitives and stale source artifacts. Parent anchors can be poorly conditioned; report minimum positive anchor dot product. Save partial evidence and unknowns after any cap, never silently renormalize mass.

## First discriminator result and frozen same-geometry mass test
Mass-disabled capture reproduces8455 gate leaves,43471 child simplices and all96 initial parents in13.46649451CPU. Every affine section conserves volume within1.44e-15. Exact dyadic checks give relative residuals−2.7362e-16(parent0),+3.1988e-16(parent1),−1.2156e-15(parent20,worst floating residual). Minimum normalized parent/leaf inequality is−2.72e-16; every anchor dot ray is at least.5. Geometry is floating, so these are necessary conservation/containment checks rather than a proof that no overlap and compensating omission exist. They provide no evidence of a volume overcount at the reported8.88e-4 scale.

Freeze the next mass discriminator now: use exactly parent pairs(0,1) and(20,21), the first ordered chamber and the chamber containing the predeclared worst section residual. Each pair joins the two nearest-facet halves of one full ordered-input chamber/sign chart. Their true Gaussian masses follow the binomial sign law:15/384 for(0,1),5/384 for(20,21). These are analytic controls independent of recurrent gates and Gaussian cone quadrature.

Evaluate the unchanged source Plackett formula on those captured leaf simplices, preserving every value and any negative output. Independently evaluate positive Duffy GL20 on the same rays, with exact dyadic radial bounds: if all pairwise ray dots are nonnegative, use min(norm²)/4 <= q <= max(norm²); otherwise use the finite15-support KKT minimum from the already-reviewed T34 radial primitive. Use its Chebyshev probability bound, excluding floating arithmetic. If a pair's summed bound exceeds1e-9, refine that same pair to32, then48 if needed; cap55CPU per phase and preserve partial status if still unresolved. Cache only model-independent rules. No clipping changes, truth fitting, random model inputs,1e7MC or full-model accuracy rerun. Any corrected full mass/reference needs a separately frozen phase after this causal check and independent review.

The first same-geometry mass launch failed at its first checkpoint because the sum of NumPy boolean negative-value flags was a non-JSON-serializable int64. Source/radial work for864 simplices ran but was not retained; the exception checkpoint repeated the same serialization failure. CPU for that failed process is unknown and additional to the rerun, not zero. `mass_failure.json` preserves the defect. Repair only casts the count to Python int; no numerical method changes.
