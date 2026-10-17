# T68: exact-rational T4 geometry reference

## Question
Can we remove floating sign, rank and topology decisions from the trained H2T4
accuracy decomposition, while preserving agreement with the existing independent
DD reference? Root claimed and pushed T68 in4f28e0e before this work. This folder
alone belongs to this task; no shared geometry or portable-cost source edits.

## Method and conceptual justification
1. Load the seed0 binary checkpoint and convert each finite floating weight with
   `float.as_integer_ratio()`. Float32 values are represented exactly by Float64,
   so this recovers their exact dyadic rational values. Carry recurrent linear
   maps and readout maps as rational matrices. This defines exact real arithmetic
   with the checkpoint weights, not the rounding behavior of a float32 forward.
2. Represent cone rays as primitive integer vectors with positive gcd removed,
   preserving orientation. Clear each rational plane's denominators by a positive
   lcm. No square-root normalization, sign tolerance or floating rank is used.
3. Lift each prefix cone using a nonzero input coefficient alpha. For an old
   integer ray r and plane a.x+alpha*y>=0, its boundary lift is
   (|alpha|r, -sign(alpha)*a.r), plus the new axial ray sign(alpha)e. Old tight
   constraints extend with a zero coordinate; all old constraints are tight on
   the new axial ray. These rays exactly generate the lifted half-cone.
4. Split a full-dimensional pointed cone by exact integer dot-product signs.
   Crossing adjacent extreme rays p,m generate (a.p)m-(a.m)p, a positive linear
   combination lying on the new plane. The common tight-constraint set identifies
   their minimal face: the pair is adjacent exactly when no third extreme ray
   contains that common set. Retain exact incidences and canonicalize new rays.
   A side containing no strictly signed ray has no full-dimensional volume.
5. After exact gate propagation, impose readout winner and second-rank planes.
   An identically zero readout margin rejects candidate j when the other index
   is smaller; otherwise it adds no constraint. Zero hidden state therefore
   belongs to output0. Input-order equalities have Gaussian measure zero.
6. Split final cones into coordinate orthants, then use pulling triangulation
   from exact incidences. Determine every face rank and simplex determinant by
   integer arithmetic. Independently audit extreme rays with signed cofactors
   of d-1 independent active planes, and check every constraint/incidence exactly.
   Within one orthant, positive L1 normalization gives rational simplex vertices.
   The sum of absolute normalized gate-simplex determinants must equal16 exactly:
   each of the16 orthant cross-sections initially has determinant1, and gate cells
   and pulling simplices partition it up to measure-zero faces.
7. Only after geometry is fixed, convert L1-normalized vertices to floating point
   for the existing positive T4 Gaussian-mass quadrature. Report this boundary
   explicitly: exact geometry does not certify floating quadrature arithmetic.

## Status
Protocol frozen before implementation and numerical work. H2T4 seed0 only in the
first cycle. Source comparison: `exact_ladder/cones.py` SHA256
c4293c925422191b17d0f4644a340bb30428e51527837711a9b5312ebd624ec6;
`small_ladder_review/t4_dd_review.json` reports54 gate cells,258 correct simplices,
accuracy .30773074302315223 at GL16, quadrature-only bound1.16788284575116e-6.
The shared source's zero-margin tie bug is not copied; the exact implementation
follows first-index argmax explicitly. No arbitrary precision silently falls
back to floating arithmetic.

## Alternatives not ruled out and falsifiers
1. A correct-looking volume can hide missing/duplicated cones or wrong incidence.
   Check exact constraints, cofactor extreme-ray certificates, full-dimensional
   ranks, nonzero determinants and exact total gate determinant16, then compare
   stage/cell/simplex counts with the independent DD reference.
2. Rational arithmetic may be too costly or suffer uncontrolled integer growth.
   Stop at110CPU seconds inside the <=120CPU cycle, <=2GBRSS, and explicit stored
   integer-bit/ray/cell limits. Record partial work and the limiting operation;
   do not silently approximate. This is a reference, not a presumed fast estimator.
3. Agreement may depend on tie handling or float integration rather than geometry.
   Include constant-class readout controls and exact zero-margin checks. Compare
   GL16/20 results while retaining their numerical-arithmetic limitation.

## Compute accounting
All runs use the primary shared flock, one OMP/OPENBLAS/MKL thread and
`ulimit -v 2500000`. Record CPU split between geometry, exact validation and mass;
integer dot products, intersections, rank/minor/determinant calls, rational map
operations, numbers of cells/rays/simplices, and observed bit maxima by object
kind. Bit maxima measured at checkpoints do not bound hidden Python big-integer
intermediates unless explicitly instrumented. No random model inputs. Model-
specific geometry and independent validation are charged separately; no free cache.

## Results
Pending.

## Claims
- [inferred] Integer projective rays and exact dyadic maps suffice for all sign,
  incidence and rank decisions in this finite cone decomposition.
- [speculative] Exact topology will reproduce the existing floating DD geometry.
- No fully certified numerical probability or compute advantage is claimed.

## Failure modes
Zero input-weight vector needs a separate lift treatment and is rejected in this
initial trained-checkpoint implementation. Complexity/bit limits stop explicitly.
Exact real-weight geometry differs from machine-rounded forward semantics on
small neighborhoods of decision planes. Floating conversion and mass integration
remain numerical until independently bounded; a quadrature truncation estimate
alone is not an end-to-end certificate.

## First bounded attempt and comparison correction
The first run built54 gate cells,221 gate simplices and257 correct simplices,
passed all exact geometry audits, and obtained determinant partition16 exactly
in0.281CPU seconds for geometry/audits. It stopped at an ordered-array incidence
comparison with floating DD; the preserved `initial_ordering_stop.json` records
the stop. The implementations order rays differently after a one-sided cut:
this reference groups signed and zero rays, while floating DD preserves order.
Pulling triangulation depends on ray order, so simplex counts can differ even
with identical cones. A permutation-sensitive comparison is therefore too strict.
Before the next run, change only the independent comparison to minimum-distance
matching of L1-normalized ray coordinates, then compare the associated exact
integer incidence bitsets. This matching uses floats to compare two outputs;
it never changes the exact construction or decides its geometry. Retain any
real coordinate/incidence mismatch and the different simplex counts. Numerical
mass follows even if floating DD disagrees; exact audits still must pass.
