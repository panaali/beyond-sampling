# T52 independent H2T5 geometry and segment review

## Question
Does the T48 H2T5 accuracy result survive an independent cone enumeration and independent checks of its positive inverse-fifth segment integral?

## Method
Use DRI `exact_ladder/cones.py` double-description gate enumeration and face-lattice pulling triangulation, distinct from T48's Qhull cross-section engine. A review-local outcome adapter handles identically zero affine margins by first-index argmax ownership, avoiding the known DD zero-margin issue queued as T60. Split resulting cones by coordinate orthants, normalize rays to L1=1, and integrate their positive T5 volumes using the T48 segment implementation. Check that segment formula independently against an arbitrary-precision antiderivative of a scalar quadratic; include equal/collinear special cases and homogeneity. Volume implementation is shared for the geometry comparison, so independence is qualified explicitly.

## Status
Running under root's pushed96c79ad T52 ownership note. Write only this review subfolder; no T48, DD, or shared evaluation edits. T47 owns portable cost accounting.

## Results
Cycle1 frozen before measurements: trained H2T5 seed0; capture SHA of every imported source and checkpoint. Compare DD accuracy with T48 committed0.2792373713559676; verify complete gate mass, determinant sum32, affine-map agreement at each gate centroid, same-orthant ray conditions, and first-index tie controls. Use target conditional integration bound1e-5, report actual bound rather than presuming target reached. No new Monte Carlo. Numerical work takes shared flock, one OMP/BLAS thread, memory cap2GB, CPU cap115 seconds; save each completed component promptly.

## Compute accounting
Record DD operation counter and CPU separately from volume planning/integration and diagnostic checks. Geometry-reference evaluation is validation, not a proposed new benchmark. All import/source and quadrature independence limits remain visible.

## Claims
[unverified] Independent DD geometry should reproduce T48's accuracy. [unverified] The inverse-fifth segment identity should match an independently derived quadratic antiderivative.

## Alternatives not ruled out
1. T48 quadrature refinements can agree despite missing/duplicated geometric regions. Cheapest independent test: DD plus pulling triangulation and full partition.
2. Shared volume-code mistakes can make two geometry engines agree incorrectly. Cheapest independent test: high-precision antiderivative checks of the eliminated coordinate and known constant-prediction accuracy1/5.
3. Geometric tolerances can hide narrow cells in both engines. Preserve numerical qualifications; compare a tighter DD tolerance in a later bounded cycle if needed.

## Failure modes
DD normalizes zero rows incorrectly without the local adapter. First-index ties can have positive Gaussian mass. Neither engine provides interval-certified geometry. The bound covers integration only. Time caps and memory restrictions must fail explicitly rather than skip material mass.

## Cycle1 results
`check.py` (with the common flock, thread and memory settings) reproduces `results.json`. Independent DD geometry gives0.27923737131338355, differing from the committed T48 result by-4.26e-11; its integration-only bound is2.72e-6. Full gate mass is1.0000000000210378 with bound2.61e-6. DD finds110 gate cells,2039 correct simplices and582 gate simplices; determinant sum is exactly32. No computed-zero determinants were omitted. Gate affine checks differ by at most1.11e-16; ray L1 errors are at most4.45e-16.

The positive inverse-fifth segment formula passes18 independently constructed checks against a100-digit quadratic antiderivative, maximum relative error7.24e-16. The reference completes the square q(u)=A*t^2+D and uses F(t)=t*(3D+2At^2)/(3D^2*(At^2+D)^(3/2)), whose derivative is q^(-5/2); equal/collinear cases use their elementary limits. It does not reuse the author's endpoint formula or quadrature. Full cycle CPU2.55 seconds; geometry0.30 seconds, accuracy planning/integration0.90 seconds, partition0.41 seconds. Operation counts are diagnostic inventories, not a separate portable-cost audit.

[measured] Independent geometry and scalar primitive checks support the original result. [qualified] The full-cone integration implementation is shared with T48 and the geometry remains floating point; this is not total-error certification.

## Cycle2 protocol
Before further measurements: tighten DD sign tolerance1e-11 to1e-13 and the requested integration bound1e-5 to1e-8 on the same original model; compare estimate, region counts, zero determinants, partition and actual achieved bound. Separately verify the review-local outcome adapter on all-zero and identical nonzero output rows, both with exact accuracy1/5. These address the known zero-margin defect without changing DD/T48 sources. No added noisy ensemble or Monte Carlo; <=115CPU seconds under the common lock.
