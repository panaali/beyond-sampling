# T123 positive-reference coverage of fixed T4 geometry

## Question
Can an independently positive mass representation validate the source-positive cones left unverified by T119, without new gate enumeration? Claim9be1457 was pushed before implementation.

## Method and frozen protocol
Use exactly6347 quadratic exact error simplices captured in T110, SHA07a0720e229bd2e0aeda511b60038efdc37f76c17aadfdeb17cf0caf3f264581. Rays are rows. For each, first test common coordinate orthant by direct min/max comparisons. Otherwise apply T119's fixed catalog: pairs lexicographic, t1/16,1/8,1/4,1/2, signs+,-; stop first common signed orthant, at most48 rotations. Catalog ignores reference values. Preserve uncovered index list; do not invent a full accuracy value for incomplete reference coverage.

For representable cones, reflect signs, normalize L1, use the previously independently checked positive GL20 tetrahedral rule. Bound each quadrature error by abs(det(Rnormalized))*(12n+9)/(6pi²*3^(n−1)); sum these conditional integration-only bounds. They exclude floating input, rotation, determinant and summation roundoff, and do not certify captured ray geometry. Record normalized determinant and conditioning diagnostics. Compare covered sums and each value with unchanged T109 source and scoped guard values; retain every discrepancy. The original source is unchanged and no reference selects nodes. No model samples or trained-parameter sweep.

First perform a bounded geometry-only inventory. Then integrate the represented corpus, checkpoint every128cones, stop at90process CPU/110hard CPU and retain partials; no nearest expansion. Full setup/rule construction and rotation work are timed. Report fixed312,000 known integrand arithmetic per represented cone separately from additional node/rule/determinant operations and full prior source work. This is independent truth validation, not a cheap production estimator.

## Alternatives and falsifiers
1. Failures could be isolated to source-negative cones: full represented-corpus comparisons test source-positive errors too.
2. A common-coordinate orthant can fail despite a valid positive representation after rotation: test the finite geometry-only catalog, preserve uncovered cones instead of declaring mass integration impossible.
3. Double determinant conditioning can dominate positive quadrature bounds: record determinant and condition, compare the existing independent MP subset, and keep exact-arithmetic qualifications. The protocol does not convert integration bounds into floating certificates.
4. Incomplete coverage cannot establish aggregate accuracy: report only represented sums and residual identities/indices. Stronger representation alternatives require a separately frozen phase.

## Status / results
Protocol frozen; no measurements yet. Independent review requested after the bounded phase.

## Compute accounting
One thread, primary lock,2GB,110CPU hard cap; at most two global numerical processes. Per-cone source calls are reused from existing artifacts. New rotations, sign tests, positive rules, integration and file/setup costs counted or explicitly partial. No full portable-cost claim.

## Claims
[inferred] Positive representation avoids cancellation between signed O(1) pieces.
[speculative] The finite coordinate catalog covers sufficient captured cones for a useful aggregate reference.

## Alternatives not ruled out / failure modes
Non-coverable cones, determinant/coordinate roundoff, expensive high-order positive integration, unresolved source-positive errors on uncovered cones, and uncertain aggregate reference precision. A finite catalog failure is not a mass-method obstruction. Broader rotations, geometric subdivision, stable Schur paths and author endpoint repairs remain available.

## Inventory result
The fixed geometry-only inventory completes .959CPU:4033 native orthants,1826 additional rotations,488 uncovered. Therefore this catalog cannot by itself establish a complete positive corpus reference. Integrating the5859 represented cones remains a valid independent diagnostic; the uncovered source sum is reported explicitly without substituting it into a purported certified aggregate. Known rotation arithmetic is1,222,656, excluding comparisons/coefficient setup.

## Covered-corpus positive reference results
All5859 represented cones complete in4.682CPU. Positive reference covered sum .46898661743868986, conditional integration-only bound sum2.62435485162e-8. Original source covered sum .4689860373789187 is lower by5.80060e-7; after negative-value fallback it is still lower by2.18694e-7. Thus the principal remaining aggregate discrepancy is accumulated source-positive errors on represented cones, subject to the explicit floating-geometry qualification. Largest source-positive discrepancy is1.73947e-8 at1459, with7.998e-14 quadrature bound; individual per-cone1e-7 tolerances need not control a network sum. Maximum represented normalized-ray condition is779629.3. The488 uncovered source masses sum .054000258566557345; these are unverified source values, not a positive reference certificate. Known integrand cost alone is1,828,008,000 arithmetic operations, unsuitable as a claimed cheap estimator.

Freeze a diagnostic independent formulation check on the three largest source-positive discrepancies, indices1459,1458,3021. Apply the separately derived all-correlation Schur path at50/70digits to original unrotated dyadic rays, retain both precisions and failures, <=95CPU soft/110CPU hard. This directly tests whether rotated positive quadrature/roundoff could explain the newly observed source-positive discrepancies. No source node choices, rotation catalog or positive reference order change.
