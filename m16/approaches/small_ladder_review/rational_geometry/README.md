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

## H2T4 seed0 result
The permutation-invariant comparison completed. Exact gate counts by step are
2,6,18,54. Every final gate cone and outcome/orthant cone used in the reference
passes integer containment, full-rank, complete-incidence and active-plane
cofactor checks. The54 floating DD gate cells have the same ray counts and
matching incidence bitsets after ray matching; largest L1-coordinate difference
is2.21e-14 and largest gate-map coefficient difference3.64e-12. These floating
comparisons do not enter the exact geometry decisions.

There are221 exact gate simplices and257 correct simplices, versus258 correct
simplices in the prior differently ordered pulling triangulation. Their exact
normalized gate determinant sum is16. The exact correct determinant sum, stored
as numerator/denominator in `results_t4.json`, is approximately4.937112672159329,
matching the independent floating reference. Equal nonzero output rows and zero
output rows generate exactly identical integer correctness cones, verifying
first-index ties beyond merely the dead-state case.

| Floating positive mass calculation on exact rays | Value | Quadrature-only bound |
|---|---:|---:|
| Accuracy, GL16 | .30773074302338094 | 1.1679e-6 |
| Accuracy, GL20 | .3077307430239267 | 1.7861e-8 |
| Gate partition, GL20 | 1.0000000000000129 | 5.7885e-8 |
| Constant-class0, GL20 | .25000000000000155 | 1.4471e-8 |

The GL20 accuracy differs from the separate Qhull reference's GL20 result by
less than1e-15. These bounds cover quadrature truncation under ideal arithmetic;
conversion, quadrature nodes/weights and floating evaluation remain unverified
arithmetic. Thus [measured] exact topology is now available for this checkpoint;
a fully certified final probability is not yet claimed.

The full successful run cost1.367CPU seconds including torch import/loading,
independent DD comparison, tie controls and all mass calculations. Gate
construction itself cost.00655CPU in the first attempt; successful timing is
stored separately in `results_t4.json`. The successful exact geometry and audits took
.23219CPU; its gate construction took.004265CPU. No large-integer limit was approached: primitive rays at most175bits,
plane normals131bits, rational map numerators132bits/denominators128bits,
tracked Bareiss intermediates893bits. The correct determinant accumulator reached
5696bits, so it is tracked separately from geometry-object4096bit limits; aggregate
rational numerators/denominators have a separate1,000,000bit cap. Ray cap2000,
cell cap20000, and the110CPU deadline remain explicit stopping conditions.

Successful-run work including controls:37315 integer dot products,1885 clips,
1140 intersections,1894 exact rank calls,17207 integer determinants,76484 Bareiss
updates,2194 cofactor ray certificates and5456 rational map products. Those counts
are algorithm-operation counts, not unit-cost big-integer FLOPs. Full integer
simplices and exact dyadic model weights are serialized in `geometry_t4.json`;
all mass conversions use those integer rays. First failed comparison cost1.247CPU
and is retained separately, so total spent across both attempts is2.614CPU.

Reproduce from the repository worktree:

```
ulimit -v 2500000
flock /workspace/work/codex-compute.lock env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/clean-neurons/.venv/bin/python m16/approaches/small_ladder_review/rational_geometry/check_t4.py
```

[measured] This checkpoint's topology needs neither tolerances nor normalized
floating rays. [inferred] The rational construction closes the prior topology
qualification for this reference, conditional on the implementation/proof audit.
Independent review is still required before a confirmed claim. Broader models,
noisy copies and fully enclosed floating mass arithmetic remain untested.


## H4T4 extension protocol, frozen before computation
Root authorized one original H4T4 seed0 case, no noisy copies. Reuse exactly the
integer gate/clip/rank/cofactor implementation and first-index tie handling, now
loading four hidden units. Keep all exact containment/incidence/partition tests,
zero/equal-nonzero readout controls and numerical GL16/20 comparisons. Reference
floating DD reports450 final gate cells,5040 correct simplices and GL16 accuracy
.47701312599404416. Triangulation count need not match under a different ray order;
compare gate ray/incidence sets after permutation matching and retain differences.

Alternatives: (1) H2 may hide non-simple or redundant-face errors; H4 increases
branch and incidence complexity, while exact cofactors/partition identity and
independent DD comparison provide falsifiers. (2) integer growth or rational
accumulation may dominate at H4; retain the same110CPU deadline,4096bit geometry
object limit,1,000,000bit rational-accumulator limit,2000ray cap and2GBRSS. Stop
explicitly, preserving partial results. Large aggregate numerator/denominator
values are serialized in hexadecimal to avoid Python's decimal digit limit.
This is exact topology for the real-arithmetic network with binary-rational
weights; normalized cone masses remain numerical. No floating fallback is added.

## H4T4 seed0 result
The same exact code completed H4T4 in10.010CPU seconds, maximum RSS273076KiB.
Gate counts by step are2,10,66,450. All exact cofactor, containment, complete
incidence and rank checks passed; the normalized gate determinant sum is16
exactly. Independent floating DD has the same gate/ray counts and incidence
sets after permutation matching. Maximum L1 ray difference2.34e-15, gate-map
coefficient difference7.11e-15. The exact pulling decomposition has3853 gate
simplices and5058 correct simplices, versus5040 correct simplices in differently
ordered floating DD. This ordering difference is retained, not treated as an
accuracy discrepancy.

| Floating mass on exact H4T4 geometry | Value | Quadrature-only bound |
|---|---:|---:|
| Accuracy, GL16 | .47701312599412177 | 1.7685e-6 |
| Accuracy, GL20 | .47701312599411855 | 2.7048e-8 |
| Gate partition, GL20 | 1.0 | 5.7885e-8 |
| Constant-class0, GL20 | .24999999999999997 | 1.4471e-8 |

The prior independent DD GL16 accuracy .47701312599404416 differs by less than
8e-14. Both zero and identical nonzero readout rows again yield exactly identical
integer correctness cones. Accuracy integration remains numerical; exact topology
is for the real-arithmetic network with the binary-rational checkpoint weights,
not a floating-point execution model.

Gate construction cost.04294CPU, exact geometry plus audits5.46517CPU, tie-control
geometry.61615CPU, independent DD comparison.06597CPU; GL16/20 accuracy plus
partition/control integration cost2.35129CPU. Remaining time includes import,
checkpoint conversion and serialization. Recorded operation counts, including
controls:556719 integer dot products,16472 clips,14219 intersections,27406 exact
rank calls,130240 integer determinants,727655 Bareiss updates,23344 cofactor ray
certificates and89920 rational map products. Total mass quadrature nodes103757568.

Geometry integer growth stayed moderate: rays306bits, plane normals134bits,
rational map numerators132bits/denominators128bits, largest tracked Bareiss
intermediate1573bits. Exact rational determinant accumulation is the growing
cost: gate partition intermediate4979bits, correct-sum numerator/denominator
149905bits. The final correct determinant sum is approximately7.47630309540539;
its full exact hexadecimal numerator/denominator is retained in the results.
This is a reference cost, not a unit-cost big-integer or efficiency-win claim.

The8.85MB raw integer artifact is stored losslessly as `geometry_h4_t4.json.gz`;
compressed size and separate post-run compression CPU are recorded in
`results_h4_t4.json`. The reproduction script now writes the compressed artifact
directly. Reproduce the previous command with `--hidden 4`. No noise sweep or
higher-dimensional geometry was attempted. Both original-checkpoint topology
references are ready for independent review; certification of floating cone
mass evaluation remains a separate unresolved step.

# T72: enclosed positive T4 mass

## Protocol and arithmetic model, frozen before implementation
Root claimed T72 inb7209a1. First H2T4 only, using the exact integer geometry
artifact; no H4 or noise sweep. The goal is an enclosure, not a nominal estimate
with a final nextafter attached. Charge the already-built exact geometry as
model-specific setup, and separately report every new rule/interval/validation
cost. One bounded <=110CPU cycle, primary flock (or approved review lock for a
<=20CPU review), one thread, <=2GBRSS.

Assume NumPy scalar/elementwise binary64 addition, subtraction, multiplication
and division use IEEE754 correctly rounded finite arithmetic, with gradual
underflow and without flush-to-zero/fast-math reassociation. Each operation is
a separate ufunc with materialized output; no BLAS dot or untracked fused
multiply-add is used. Assume nextafter returns the adjacent representable value.
Record NumPy/platform versions and test tie rounding/subnormals. These runtime
tests are falsifiers, not a formal proof of hardware/compiler compliance. Stop
on nonfinite values, invalid intervals, nonpositive reciprocal denominators, or
failed rational containment. The certificate is conditional on this explicit
arithmetic model and the exact-geometry proof/implementation audit.

Use GL20 tensor quadrature. NumPy leggauss supplies guesses only; bracket each
Legendre root with rational dyadic endpoints, verify exact opposite polynomial
signs, and bisect to width<=2^-96. Twenty disjoint sign-changing intervals inside
(-1,1) account for all twenty simple roots of the degree20 polynomial. Enclose
weights by rational interval evaluation of
w_i=2(1-x_i²)/(n² P_(n-1)(x_i)²), using the root identity P_n(x_i)=0. Enclose pi
with the rational alternating-series Machin identity
pi=16 atan(1/5)-4 atan(1/239). Every rational-to-binary64 endpoint conversion is
checked against its exact Fraction and expanded outward when needed.

For nonnegative intervals, round each product/addition/division outward, and
evaluate1-u as [down(1-u_upper),up(1-u_lower)]. A lower bound may be raised to0
only when nonnegativity is already proved. Convert each exact L1-normalized
same-orthant ray to its absolute coordinate intervals; reflection preserves
Gaussian mass. Build barycentric coordinates and Jacobian-weight intervals,
then enclose squared norms, reciprocals and integrand products operation by
operation. Use explicit outward pairwise reductions for nodes and simplices.
Use the exact rational determinant from the geometry artifact, not linalg.det.

The analytic truncation bound is also enclosed. On a same-orthant L1 simplex,
1/4<=||Vy||²<=1. The Chebyshev approximation to s^-2 of degree n-2 has uniform
remainder E=(12n+9)/3^(n-1). Under the stick-breaking map, its polynomial times
the Jacobian has degree<=2n-2 in each cube coordinate, so GLn integrates it
exactly. Positive quadrature and integral both give Jacobian mass1/6; therefore
mass error<=|det V| E/(6 pi²). All constants/determinants here are rational except
pi, whose interval is explicit. Add this bound outward to the quadrature interval.

Independent containment falsifiers: (1) arithmetic assumptions or weight/root
bracketing may be wrong. Check fixed adversarial binary64 operations against
exact Fractions, all polynomial GL moments through degree39, and the tensor
Jacobian mass1/6. (2) apparently narrow intervals may omit an operation or use
an unjustified lower bound. Check the coordinate-orthant mass1/16, compare fixed
cone quadrature values against independent high-precision evaluation, and ask
T55 for a separate source/containment review. Keep raw interval widths and every
failure; do not select whichever bound agrees with the old floating estimate.
The exact H2T4 gate partition and constant-class controls remain available for
later extension, but the first full numerical enclosure covers its257 correct
simplices only. No final certified-probability claim before independent review.

## T72 first enclosure result, pending independent containment review
The first H2T4 run completed in2.840CPU seconds, maximum RSS38316KiB. Rule setup
cost1.517CPU: rational root/weight/pi setup.2954CPU and the40 exact rational
moment checks1.2174CPU. All moment degrees0..39 were enclosed, with largest
rational interval width1.763e-22. The tensor Jacobian integral interval
[.16666666666666607,.16666666666666732] contains1/6. All136 fixed elementary
arithmetic cases contained exact Fraction answers; tested subnormal and tie
rounding behavior matched the stated assumptions.

The257 correct simplices required1.299CPU and2,056,000 GL20 nodes. Their exact
quadrature value, conditional on the arithmetic model, is enclosed by
[.30773074302392284,.3077307430239303], width7.44e-15. Adding the enclosed analytic
truncation bound1.7861484466067163e-8 gives the probability interval

**[.3077307251624383, .3077307608854148]**.

Its width is3.5722976465812906e-8. This is an enclosure claim conditional on the
frozen IEEE/NumPy assumptions and exact geometry proof; independent T55 source
and high-precision containment review is still pending. The old floating GL20
value lies inside the quadrature interval but was not used to select it.

The orthant control's GL20 quadrature interval is
[.06250000000000085,.06250000000000214]; it does not contain the true1/16 because
the finite quadrature has bias. Its total interval after truncation,
[.062499996382201196,.0625000036178018], does contain1/16. Keeping these two
intervals distinct is essential; directed rounding alone does not bound
quadrature approximation error.

Full integration work, excluding rule setup and orthant control:86,353,028
scalar multiplications,86,452,218 additions,4,112,514 divisions, two final signed
add/subtract operations and176,917,762 nextafter operations. These include both
interval endpoints and outward pairwise reductions. There are4112 exact ray
coordinate conversions. Counts omit Python loop/bookkeeping and rational
operation bit costs, which are recorded separately by rule/geometry counters;
no audited FPE or equal-compute claim is made. Full setup included1180 exact
Legendre evaluations,1140 bisections,22420 exact recurrence steps,360 rational
interval recurrence steps,800 moment terms and one untrusted seed-rule call.
Already-charged exact model geometry must be added to any cold estimator cost.

`rule20_certificate.json` stores exact rational root/weight/pi brackets;
`enclosed_h2_t4_results.json` stores every simplex quadrature interval and
truncation upper bound. Run `check_enclosed.py` under the same primary flock and
single-thread command. Root independently replayed the H2 and H4 exact artifacts
with a separate determinant expansion and integer forward calculation (b1cb920),
providing additional geometry support without independently proving the complete
partition algorithm. Original exact artifacts remain unchanged.
