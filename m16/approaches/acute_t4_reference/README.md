# T126 acute-ray positive reference

## Question
Can coordinate-free positive integration validate T123's488 uncovered cones without new gate geometry? Claim2953733 was pushed first.

## Method and frozen protocol
Use only the488 fixed uncovered indices from T123 and original6347 quadratic rays, SHA07a0720e229bd2e0aeda511b60038efdc37f76c17aadfdeb17cf0caf3f264581. Determine all six pairwise ray-dot signs exactly using Fraction representations of stored binary64 coordinates. A strictly negative pair dot proves no orthogonal transformation can place those original rays in a shared coordinate orthant: dot products are invariant and common-orthant dot products are nonnegative. Preserve all such cases; this excludes only that representation.

For all pair dots>=0, normalize each ray by its positive L2 norm. The Gram matrix has diagonal1 and nonnegative entries. For barycentric y>=0, sum y=1, q=||sum y_i r_i||²>=sum y_i²>=1/4 and q<=1 by the triangle inequality. The general conic change x=r sum y_i r_i has Jacobian abs(det R)*r³ for any independent rays, irrespective of orthant. Therefore Gaussian cone mass is abs(det R)/(2pi²) times the positive simplex integral of q^-2. T51's unchanged Chebyshev polynomial/Duffy degree argument yields the same GL20 integration-only bound abs(det R)*(12n+9)/(6pi²*3^(n−1)). T55 has independently checked this extension. Floating square roots, determinant, nodes and inherited geometry remain outside this bound.

Before trained runs, check independent block-pair orthant products for correlations in{-0.9,-0.4,0} each. Analytic inverse-Cholesky ray matrices have nonnegative pair dots; test base, positive row scaling2±20, and a fixed rational orthogonal rotation producing mixed coordinates. Compare fixed GL20 to product[1/4+asin(a)/(2pi)]*[1/4+asin(b)/(2pi)], permitting only the integration bound plus5e-14 floating control tolerance. Include a constructed negative-dot cone as representation exclusion control, not invalid Gaussian input.

Then perform exact signs on all488 and integrate the acute subset with fixed GL20, no tuned order. Preserve nonacute indices, per-cone results, absolute/covered-source errors, summed conditional bounds and setup/integrand/rotation costs. Partial checkpoint every64cones;20CPU control/sign phase and<=110CPU numerical phase, one thread/locks/2GB. No nearest expansion or repeated original geometry. If coverage is incomplete, do not claim an aggregate certificate; remaining geometry could admit general q lower bounds or subdivision in a separate frozen phase.

## Status and results
Protocol frozen; implementation pending.

## Compute accounting
Fraction/sign work, normalization, nodes/rules, determinant and summation all timed. Known39 arithmetic operations per integrand node is a partial lower-level convention, not full portable cost. This is reference validation, not a cheap estimator. Previous T123 cost remains separately visible.

## Claims
[inferred] Acute unit rays suffice for the positive T51 q-range without coordinate-orthant representation.
[inferred] Negative pairwise dot obstructs all orthogonal common-orthant transforms, without obstructing other positive representations.

## Alternatives not ruled out and failure modes
Floating normalization/determinant error; costly positive quadrature; broad cones with negative pairs; insufficient precision of the old aggregate reference. General radial lower bounds, alternate cross sections, triangulation, or stable source formulas remain open. Finite coverage failure is not a Gaussian mass-method negative.
