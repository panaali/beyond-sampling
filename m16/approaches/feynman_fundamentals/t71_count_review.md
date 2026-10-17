# T71 counted composition: independent T55 review

Read-only review, 2026-10-17. No estimator code changed and no numerical
integration rerun. The scope is the new fixed-tangent integration and its
composition with previously reviewed T52 geometry, determinant, Gaussian-rule,
Bernstein-bound and segment code. It is not a fresh topology audit of T52.

## Finding

The fixed-dimensional arithmetic allocations are conservative under the
declared symbolic scalar-operation model, provided fixed small integer powers
are expanded into multiplication operations. General noninteger powers, square
roots and atan2 retain the declared charge20. This convention should be explicit
in COUNTING.md: interpreting every syntactic NumPy power as a charge20 call
would make some individual line-group explanations, notably outer-rule setup,
ambiguous. This is an operation model, not a machine-instruction upper bound.

No uncharged QR/SVD, gamma function or opaque geometry kernel was found on the
counted integration path. The passed frame_builder callback uses two-pass MGS;
the passed fallback callback uses the counted rule/segment engine. The optional
QR/SVD path in fixed_tangent.py is not executed by counted.primitive.

The reviewer finds no arithmetic or quadrature-composition issue requiring a
new model run. The supplied full-estimate counts remain proposed portable
cost bounds subject to parent/DRI acceptance of the convention and inherited
T52 accounting. This does not certify floating geometry or numerical error.

## Fixed frame and line groups

Each width5 dot costs9 scalar operations. The two-pass MGS loop has exactly
2*(0+1+2)=6 projections. Each projection/update costs9+10=19 arithmetic units;
the charge27 leaves8 for tests/control. Each normalization costs9+20+5=34
before tests; charge48 is conservative. The3x3 Gram product costs81, with9
subtractions,9 absolute values,8 max comparisons and a final comparison/boolean
operation comfortably within128. Together with64 control units, the explicit
frame charge is6*27+3*48+128+64=498 per cone.

The non-frame per-cone groups total896. The principal coordinate operations
are15 edge differences, a5-by5 times5-by3 product135, a2-by3 times3-by5 product50
plus10 residual subtractions, and9 further edge differences:219, below256.
Seven width3 cross products cost63;24 face-edge differences,20 face-normal
norm arithmetic and5 determinant arithmetic bring that group to112, below128.
Validation and inverse/control allocations are ample for their fixed shapes.

All outer nodes, including eventual rejections, are charged1472 units:
128 endpoints,384 norm/conditioning,448 facets,384 radial/screen,128 control.
Endpoint interpolation has15+36+60=111 arithmetic operations. Norm/condition
work includes9 for delta squared,15 for the3x3 inverse-vector product, fewer
than20 for the inverse-Frobenius expression,39 for the20-coordinate squared
norm, fewer than30 for the condition expression including sqrt,116 for four
width5 endpoint norms, and20 unit-coordinate divisions; these fit384.

Each facet's112 allowance covers a dot5, cofactor norm23 including sqrt20,
angle numerator at most5, three width5 pair dots plus additions30, atan2 and
doubling21, and the signed quotient plus mask/control. Radial support bounds
need about202 arithmetic/transcendental units including their sqrt; allowance
220 is sufficient. The surrounding384 radial/screen group also covers two
noninteger powers40, cancellation ratio, denominator and range comparisons.
The additional128 per-node control covers selections, acceptance statistics,
signs, scalar loop work and reductions not already itemized.

Thus the integration's pre-fallback charge is1394 per cone plus1472 per outer
node. This excludes separately counted determinants, rule generation, geometry,
bound planning and fallback work; none of those is inferred from wall time.

## Positive fallback

The finite product in tail() computes (5/2)_N/N! and r^N without gamma calls.
Each loop needs five arithmetic operations plus control, within charge8.
Setup100 covers two square roots40 plus ratio/scale algebra. Finish80 covers
two additional square roots40 and the positive denominator arithmetic. The
result is exactly the previously proved bound

    2*C^(-5/2)*(5/2)_N/N!*r^N
      / ((1-r*(N+5/2)/(N+1))
         *(1-r^2*(N+5/2)/(N+1))^(5/2)).

The ratio>=1 branch returns an unavailable bound and the caller advances the
order; it does not silently accept the rule. Every attempted bound is charged.
The cached positive rule is local to one estimate and its cold construction is
charged. The fallback rule's32 per tensor point covers Duffy coordinates and
weights. Endpoint construction is35 arithmetic units, below48. The inherited
I5 segment needs27 for three width5 dot products, roughly16 further arithmetic
units, and two square roots40, below its charge90. Weighted reduction2 per
node plus32 control is adequate. The fallback radial bound256 is also above
its fixed-shape work. Rejecting a node still pays the full analytic work before
this fallback is charged.

## Degrees, measure and positive error allocation

For the T5 original four-dimensional parameter simplex, a degree-K polynomial
in q=||Vy||^2 has degree at most2K in each Duffy coordinate. The first-coordinate
Jacobian has degree3, so GL24 exactness through degree47 allows K=22. Applying
an exact positive inner tetrahedron integral preserves this argument. The
inherited Bernstein bound uses precisely n-2=22, and its normalization is
unchanged by analytical elimination of the last three coordinates.

For the fallback's tetrahedron parameter domain, the remaining two numerical
Duffy coordinates have Jacobian degrees2 and1. Thus order n permits K=n-2,
as implemented. The exact positive I5 segment integral preserves the degree
argument. Both the true and positive quadrature remainder masses have factor
volume(Delta3)=1/6, hence uniform-tail/3 is correct.

Outer weights are positive and sum to1/4 in exact arithmetic. For cone factor
f_i>0 and N cones, inner tolerance B/(N*f_i*sum(weights)) allocates at mostB/N
after weighted integration. Summing across cones gives at mostB. The recorded
weighted bounds can be much smaller than the allocation. Ordinary floating
rounding of these quantities is not covered by the real-arithmetic proof.

## Generalized affine-plane validation

The moment identity requires a nonzero affine plane separated from the origin,
not exact coordinatewise orthant signs. The revised check on a shared signed
normal is therefore mathematically appropriate for the observed tiny DD
wrong-sign coordinates. No ray modification is needed. For a support direction
n, only max(0,min_i n.dot(v_i))^2/||n||^2 is a valid lower bound; squaring a
negative minimum would be invalid. The revised radial_interval correctly
clamps before squaring.

The input residual threshold and clipping lower<=upper by one rounded ulp are
numerical domain policies, not directed enclosures. They do not upgrade the
inherited topology/roundoff status. The documented tiny wrong-sign extrema
must remain in the result provenance. A generalized API allowing zero radial
lower bound or antipodal endpoints would require additional failure handling;
the reviewed T5 signed-plane construction stays separated from the origin.

## Interface and exclusions

The public design is fixed GL24; budget is advisory and rng unused. It charges
the complete calculation on each call, including local fallback rule setup.
The affine-only sampling denominator65 is the inherited conservative H2T5
comparison, which gives sampling free input generation, gates and ranking.
This number is not a generic forward cost for arbitrary H,T. Source loading,
memory allocation/copies and Python dispatch follow the declared T52 exclusions.
Measured CPU and model preparation remain separate reporting quantities.

The adapter keeps COST_REVIEWED=False pending acceptance. This review supplies
independent support for its integration accounting but does not change that
flag or authorize a global solved claim. Existing geometry and floating-error
qualifications must remain visible when a finite cost is enabled.

## Reviewed sources

Paths are relative to the T71 odd_tetra folder in recover-small_h2_56.

| File | SHA256 |
| --- | --- |
| counted.py |343b384da8acc420773e7e6f47f75a10bf6acaef3702e4837618cca0f1579b6b|
| fixed_tangent.py |b80e7f28b1f81a9f8bd0d4efded403214cb37f8c120d357da9022528ed0b90b3|
| primitive.py |a537f029e671fe26827364cfbaff092daeffe42c07d5f9e175aabec3dc457173|
| estimator.py |2a571b54c8c003f8bb2169370819cf83cb877433f47fdfe16b3423743f4dfa2a|
| COUNTING.md |cb1a9b3f51ffe56567cbe73c9713d1ba95b02d332075165b6cec5a8d82ebdc3a|
