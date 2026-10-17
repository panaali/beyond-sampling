# T130 cost-first positive cone masses

## Question
Can the positive representation independently validated in T123/T126/T129 achieve useful absolute error with inexpensive geometry bounds and short rules? Claimd3b45e9 was pushed before implementation. This is a distinct representation study, not an edit to T109's single-edge source.

## Method and frozen protocol
Use original6347 quadratic rays and their full positive references, but never use saved references to choose outputs/orders. For original rays r_i choose c=sum r_i, m=min_i(r_i dot c). If m>0 then a=m²/(c dot c) is a valid lower bound on q=||sum y_i r_i||² by Cauchy-Schwarz; b=max_i||r_i||² is an upper bound by convexity. If m<=0 or determinant vanishes in floating arithmetic, report unsupported/roundoff ambiguity, not zero probability. Compute determinant by explicit24 signed Leibniz products (95 scalar mult/add operations), avoiding an uncharged LAPACK inverse/determinant proxy. Orientation is removed only with absolute determinant.

For D=abs(detR)/(12pi²), true mass lies in[D/b²,D/a²] in exact arithmetic. Return its midpoint when half-width meets the requested tolerance. Otherwise use the general positive Chebyshev/Duffy rule bound from T129 and choose first n in{2,3,4,6,8,12,16,20,24,32} meeting that tolerance. If none qualifies retain unresolved, including all attempted selection work; do not silently use an inaccurate value. Geometry/selection arithmetic is instrumented by actual scalar operations; transcendental calls and comparisons counted separately. Positive integration uses39n³ known scalar operations plus final scale; GL construction/setup timings and omissions stay explicit.

Freeze two separate tolerances:1e-7 per cone (author primitive goal) and1e-7/6347 per cone (sum-budget goal). This is a comparison of two stated objectives, not a threshold fit. Precede trained runs with27 analytic block orthant/scale/permutation controls from T129 at both tolerances, exact rational radial-bound controls on a fixed16-case subset, and explicit unsupported/degenerate fixtures. Every raw failure is retained. Positive/negative source discrepancies do not alter the recipe. Run each entire corpus cold, checkpoint128cones,90CPU soft/110CPU hard or stricter20CPU review cap. No nearest/recipe/noise or neural enumeration expansion.

Rules are deterministic universal functions of order; an in-call cache avoids repeated construction, starts empty for each corpus, and cold setup is timed. Universal constants/permutations are model-independent; no model-specific advice cache or saved-value lookup. Complete CPU includes implemented setup, but portable GL-construction arithmetic remains unaudited and cannot be called free. Compare only named arithmetic with like qualification, never relabel partial counts as portable total.

## Alternatives and falsifiers
1. Thin-cone robustness could cost far more than source masses: preserve actual rule/order distribution, selection cost and failed calls.
2. Cheap center bounds could be too loose despite well-behaved exact q minima: preserve unsupported/oversized cases; exact KKT or alternate centers are separately reviewable rescues, not automatic escalation.
3. Explicit determinant arithmetic could lose tiny masses: compare independent full references and exact selected determinants; no floating certificate follows from an analytic bound.
4. Meeting1e-7 per cone need not control a6347-cone sum: report the second fixed aggregate objective separately.

## Status / results
Protocol frozen; no measurements yet.

## Compute accounting
Named scalar arithmetic/comparison/transcendental counts correspond to executed setup/selection expressions, plus39n³ integration arithmetic. GL construction, memory/vector-library internals, floating guarantees and benchmark hardware differences remain limitations; complete CPU and cold cache events are reported. No sampling/MSE, cheap-primitive or solved-rung claim before independent review.

## Claims / remaining alternatives
[inferred] Separating-center bounds produce positive mass intervals without signed cancellation, conditional on exact arithmetic.
[speculative] Interval shortcuts and small positive rules can reduce useful mass cost. Stronger geometric bounds, exact moments, stable determinants, subdivision and author endpoint repairs remain open if this finite variant fails.

The exact-rational setup subset is the same predeclared16 quadratic indices used by T119:0,2115,4231,6346,3222,3219,3221,3224,5962,3251,3254,3836,2567,2577,5448,2020. It is not selected from new cost-study results. Known node-plan arithmetic is additionally18n³+3n per cold order for the implemented Duffy-coordinate/weight expressions; Legendre-rule generation remains separately timed/unaudited. Direct dot-product integrand count is39n³+1 including the final mass scale and exact reduction length; this refines the earlier39n³ convention.

## Control results
Of54 fixed analytic objective cases,15 return values and pass their stated bound plus5e-14 floating allowance. The other39 explicitly fail the finite order target; they are retained as precision/cost limits, not relabeled passing or converted to zero. All16 fixed exact setup cases have valid positive center; largest explicit determinant relative deviation1.776e-11. Degenerate/nonseparating fixtures are rejected explicitly. These controls validate returned formulas, while disproving universal coverage of this frozen finite-order recipe. Proceed with the already-frozen trained corpus to measure its actual coverage and cost; do not expand the order set in response.
