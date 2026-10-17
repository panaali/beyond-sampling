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
