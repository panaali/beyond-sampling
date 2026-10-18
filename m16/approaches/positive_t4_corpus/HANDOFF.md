# T123 bounded positive-reference coverage handoff

Claim9be1457 preceded implementation. Exact same6347 quadratic T110 rays as T119; no neural enumeration or source changes.

The frozen48-candidate rational rotation catalog covers1826 cones in addition to4033 native coordinate orthants. T55 independent Fraction audit868c69b verifies exact shared orthant signs for all5859 transformations of stored dyadic rays;488 remain uncovered. Positive GL20 references for5859 cones finish4.682CPU. Their covered mass is .46898661743868986; summed integration-only error bound2.62435485162e-8. Geometry, determinant and floating quadrature error are outside that bound. Condition maximum779629.3 is recorded, not converted to a rigorous roundoff estimate.

Original T109 covered mass sum is lower by5.8005977118e-7. T119's negative/exception-only fallback still leaves covered mass lower by2.1869397546e-7: accumulated small errors on positive source outputs matter even when individual1e-7 requirements are met. Largest three source-positive discrepancies at1459/1458/3021 are1.73947e-8,−1.62070e-8,+1.01505e-8. Independently implemented unrotated-ray MP Schur references at50/70digits confirm all three; positive reference differs by at most3.56e-20. Both precision levels and all original failures are preserved.

The488 uncovered source masses sum .054000258566557345, but this is not independently verified coverage. Combining that unverified residual with positive covered values gives a diagnostic accuracy .4770131239947528,−2.00e-9 from the independently obtained whole-model .477013125994119. This close agreement cannot be called a full positive certificate. No uncovered residual is silently dropped or replaced with zero.

Costs: geometry inventory .959CPU plus1,222,656 known rotation arithmetic, excluding coefficient/sign/setup operations; covered reference integration4.682CPU and1,828,008,000 known integrand operations, excluding rules/normalization/determinants/file work. Full CPU includes these performed steps. This is expensive independent truth validation, not a competitive estimator. No nearest corpus expansion was performed.

Next useful representation rescue: unit-L2 rays with pairwise nonnegative dot products imply1/4<=||Σy_i r_i||²<=1, even without a common orthant. If independently accepted, the same Chebyshev/Duffy bound would apply directly. Negative pairwise dot products prove that no orthogonal common-coordinate-orthant transform exists for those rays, but do not obstruct positive integration with other normalizations, subdivision or general radial bounds. These alternatives remain open; the finite rotation catalog is not a family-wide negative.

Reproduce modules `inventory`, `integrate`, `check_mp`, under recorded one-thread locks and CPU/memory caps. Saved results are complete; no rerun needed for inspection. Root/DRI registry requested. T55 final numerical source review pending; no self-confirmed claims.
