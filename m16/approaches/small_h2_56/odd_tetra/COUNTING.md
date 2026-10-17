# Counted T5 tetra composition

The scalar-unit convention follows T52: addition, subtraction, multiplication,
division, absolute value, comparison and sign/control tests cost one; sqrt,
atan2 and other elementary transcendental calls cost20. Memory allocation and
copies are excluded in both algorithms. This is an operation model, not CPU
time, bit complexity or a floating-error guarantee.

The counted engine imports the unchanged T52 bounded-rank DD geometry,
determinants, Gauss-Legendre Newton setup and positive I5 segment. Their counters
remain active. No QR, SVD, gamma, Qhull or other opaque numerical kernel is used
in the counted integration. The source engine still supports the original QR
path, but `counted.primitive` supplies the explicit frame callback and fallback.

For each three-column tangent frame, two-pass modified Gram-Schmidt has six
projection/update pairs. Each pair needs a width5 dot (9) and subtraction of a
scaled column (10);27 units include loop/test allowance. Three normalization
steps each cost at most48 (dot9, sqrt20, five divisions and checks). The3x3
orthogonality product needs81 operations;128 covers its subtraction, absolute
value, maximum and comparison. A further64 covers frame initialization/control.

The remaining vectorized `fixed_tangent.py` code is charged by the following
line groups, even for rejected nodes. The stated allocations exceed their
fixed-dimension arithmetic counts; no observed timing enters them.

| Group | Charge | Covered arithmetic |
| --- | ---: | --- |
| Per-cone validation |256|25 finite tests, common-orthant extrema/tests, signed L1 normal and residual validation|
| Per-cone coordinates |256|15 edge differences; tangent product135; residual product/subtraction60; nine tangent-edge differences|
| Per-cone cross products |128|Seven3D cross products63; face-edge differences24; face-normal squared norms20; edge determinant5|
| Per-cone inverse/control |256|Nine inverse entries, inverse norm17, orientation/rank branches, index and batch overhead|
| Per-node endpoints |128|Perpendicular interpolation15; tangent interpolation36; ambient interpolation60; remaining scalar interpolation setup|
| Per-node norm/condition |384|Perpendicular norm9; inverse-times-tangent15; inverse Frobenius expression<20; ambient squared norm39; conservative condition expression<30; four endpoint norms116; unit-vector divisions20; tests/control|
| Per-node facets |448|Four facets at112 each: dot5, cofactor norm<24, angle numerator<8, three width5 pair dots and sums30, atan2 and factor21, signed term<8, loop/test allowance|
| Per-node radial/screen |384|Supporting-plane radial interval<220 including sqrt; two endpoint norm powers40; cancellation ratio<20; final denominator/sum<20; positive range tests/control|
| Per-node control |128|Remaining comparisons, sign/abs, selection/mask handling, loop arithmetic and conservative slack|

Each fallback separately charges radial interval256, every bound trial, finite
rising-factorial product, all Gaussian rule setup and Duffy nodes. Its endpoints
cost at most48 per node (matrix product15, endpoint additions/multiplications20,
control); imported I5 charges90 including two sqrt; weighted sum charges2 per
node plus32. The rule cache is local to one estimate, so cold setup is counted.
The exact fallback constant is tetrahedron volume1/6; two positive remainder
errors give uniform-tail/3. GL degree n-2 accounts for the largest Duffy
Jacobian exponent2.

Outer setup, positive inner allocation, model geometry, all5x5 determinants,
bound planning and weighted sums are charged separately. The fixed outer24
rule is a design constant; references and validation costs are not estimator
advice. Saved checkpoints/noisy model construction and source loading are
reported separately as input preparation, consistent with T52's provided-model
interface. No rounding or topology certificate is implied by a finite count.
