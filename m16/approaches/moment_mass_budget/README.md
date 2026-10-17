# T135 analytic moment intervals for positive cone masses

## Question
Can exact low-degree simplex moments replace enough expensive tensor quadrature to rescue T130's cost failure? Claimad9c2c9 preceded implementation. T55 independently enumerated all256 fourth-degree Dirichlet monomials and matched55 Gram-polynomial coefficients before numerical interpretation.

## Method and frozen protocol
Same6347 original quadratic rays, same two tolerances1e-7 and1e-7/6347, same cold-rule caches, explicit determinant and scalar-cost convention, and same finite GL orders as T130. Preserve the original interval shortcut first. Only when it fails, calculate exact-in-real-arithmetic moments of q=yᵀGy for uniform y on the3-simplex: s=sumG, t=trG, r=G1,d=diagG,
`mu=(s+t)/20`,
`E q²=[s²+2st+4 sum r_i²+t²+2 tr(G²)+8 sum d_i r_i+6 sum d_i²]/840`.
Because q>=a, Jensen and Taylor with f''(q)<=6/a⁴ give
`mu^-2 <= E[q^-2] <= mu^-2+3 Var(q)/a⁴`.
Scale by abs(detR)/(12pi²), intersect with the original mass interval, and return midpoint only if its half-width meets the unchanged tolerance. Otherwise use the unchanged finite positive quadrature fallback. Reuse geometry's diagonal/row-sum/center quantities; calculate only six extra offdiagonal dots. Charge every moment and failed shortcut operation.

Floating variance is computed by subtraction and can understate even when positive; there is no floating certificate. If computed variance<=0 or intervals reverse, preserve an explicit failed-moment diagnostic and do not use its shortcut; the independently bounded original positive-rule fallback may still return. Never clamp variance to zero or alter tolerance. Exact moment polynomials on the fixed16 prior setup cases and54 analytic controls precede trained runs. Original T130 outputs/costs remain the frozen baseline, not tuning data. Two full finite objectives only; no new models/noise/geometry, no32-order extension or source edits.20CPU review/110CPU primary cap, partial checkpoints128cones.

## Alternatives and falsifiers
Extra moment setup could outweigh saved quadrature; report total failed and successful work. Center lower bounds could remain too loose on broad cones; preserve finite-order unresolved cases. Floating variance/determinant cancellation could create spuriously tight intervals; compare independently positive references and exact moment controls and retain all diagnostics. A negative result applies to this formula/order/cost convention, not every positive or analytic mass method. Higher exact moments, better lower bounds, stable variance forms, or alternate integral representations remain open.

## Status / results
Frozen protocol; implementation pending.

## Compute accounting
Same explicit scalar Counter,95-operation determinant,39n³+1 kernel, cold18n³+3n node plans, and ordinaryπ-constant charges as T130. Legendre-rule generation is timed but portable work unknown. Complete CPU includes moment/setup/fallback costs; no production oracle lookup. Any saved-array comparisons are evaluation-only.

## Claims / failure modes
[inferred] The moment interval is sound in exact arithmetic and potentially sharper than a radial-range interval.
[speculative] Enough shortcuts will pass to improve cost. Accuracy bounds remain conditional on exact geometry/arithmetic, with floating variance cancellation explicitly unresolved rather than silently certified.

## Results and bounded completion
Moment shortcuts replace1576/1599 quadratures at the two frozen objectives but save only3.48%/0.544% named work, leaving86.395M/362.967M operations under the declared partial convention. Per-cone total accuracy error−8.80e-6 persists; aggregate objective still has the same32 unresolved cases and no full estimate. All returned values pass their tested bounds, but exact16-case controls expose4.38% relative error in a tiny floating variance, so no roundoff certificate is claimed. Higher-cost rule tails dominate; no broad method negative follows. [HANDOFF.md](HANDOFF.md) records comparison details and remaining alternatives.
