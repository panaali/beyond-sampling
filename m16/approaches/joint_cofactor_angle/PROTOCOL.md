# Frozen joint-angle gate

Claim d7ad9e62, existing Ready T109; independent theorem discussion with T55 and56. This phase first checks exact stored-matrix algebra and angle enclosures on twelve fixed analytic cases, then separately gates the original64 archived nodes selected in frozen/minors_manifest.json. No source mass, model, geometry, MC or native determinant calls.

The exact target for symmetric positive-definite binary64 Rt is atan2(-cofactor_kl, sqrt(det(Rt)*det(Rt[{i,j},{i,j}]))), where k,l complement i,j. Jacobi gives radicand=cofactor_kk*cofactor_ll-cofactor_kl^2. The original source uses floored native det(Rt) times its separately rounded om; even exact determinant cannot remove nonunit diagonal/om drift. Target angle, stored source arguments and source approximation must remain separate.

A preliminary proposal confused the asin denominator sqrt(cofactor_kk*cofactor_ll) with the atan2 denominator. Source reading and independent reviewer checks caught it before code/execution; corrected target above is frozen. No result used the wrong identity.

Exact Fraction permutation determinants, symmetry and leading-principal-minor positivity gate support. Singular/indefinite/asymmetric cases remain explicit unknown; no flooring. Positive radicand is required. Reuse reviewed rational sqrt/atan engine with absolute160-bit square-root brackets; if lower sqrt iszero, return unknown_precision rather than invent a positive lower. Round ratio interval outward to160 dyadic fractional bits before atan, controlling Fraction growth.64 alternating terms and Machin pi are inherited, with operation counts and integer bit lengths. This is a slow exact discriminator, not a cheap candidate or native speed proof.

Analytic twelve cases: identity for all6 complementary pairs; correlated block(2,3) with rho=+1/2 and-1/2 for pair(0,1); diagonal drift at entry0 by2^-52 on positive block; singular rho1; indefinite rho2; asymmetric single offdiagonal. Expected supported9/unknown3. Supported controls independently verify Jacobi and high-precision inverse partial-correlation angle; exact target enclosure depends only on rational bounds, high precision is diagnostic. Every attempt durable. No adaptive fixtures or discarded failures.

Alternatives: source error could be numerator cancellation (test exact numerator versus stored source numerator); denominator/diagonal discrepancy could dominate (hold numerator exact and compare source denominator). Small enclosure widths would not imply small total cone-mass error; quadrature, covariance construction, coefficients, source summation and libm remain outside. All64 application scope requires a separate source/pin gate after analytic controls.

One thread,2GB, review lock<=20CPU. Refuse existing outputs. Preserve exceptions and failures. Exact arithmetic/setup costs are not converted to a scalar/native budget.
